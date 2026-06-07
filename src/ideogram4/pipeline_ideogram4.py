from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from posixpath import dirname as _posix_dirname, join as _posix_join
from typing import Optional, Sequence, Callable

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from ideogram4.autoencoder import (
  AutoEncoder,
  AutoEncoderParams,
  convert_diffusers_state_dict,
)
from ideogram4.caption_verifier import CaptionVerifier
from ideogram4.constants import (
  IMAGE_POSITION_OFFSET,
  LLM_TOKEN_INDICATOR,
  OUTPUT_IMAGE_INDICATOR,
  SEQUENCE_PADDING_INDICATOR,
  QWEN3_VL_ACTIVATION_LAYERS,
)
from ideogram4.latent_norm import get_latent_norm
from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from ideogram4.quantized_loading import (
  FP8_TEXT_ENCODER_CONFIG_FLAG,
  is_comfy_quant_state_dict,
  is_bnb4bit_state_dict,
  is_fp8_state_dict,
  is_nvfp4_state_dict,
  load_comfy_quant_state_dict,
  load_bnb4bit_state_dict,
  load_fp8_state_dict,
  swap_linears_to_comfy_quant,
  swap_linears_to_bnb4bit,
  swap_linears_to_fp8,
)
from ideogram4.scheduler import (
  LogitNormalSchedule,
  get_schedule_for_resolution,
  make_step_intervals,
)


def _load_log(message: str) -> None:
  print(f"[ideogram-load] {message}", flush=True)


def _load_safetensors_state_dict(path: str) -> dict[str, torch.Tensor]:
  """Clone tensors immediately to break safetensors mmap backing."""
  return {name: tensor.clone() for name, tensor in load_file(path).items()}


def _normalize_state_dict_prefix(
  state_dict: dict[str, torch.Tensor],
  anchors: tuple[str, ...],
) -> dict[str, torch.Tensor]:
  """Strip common wrapper prefixes from repackaged single-file checkpoints."""
  if any(anchor in state_dict for anchor in anchors):
    return state_dict

  for prefix in (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
    "text_encoder.",
  ):
    if any(f"{prefix}{anchor}" in state_dict for anchor in anchors):
      return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
      }

  return state_dict


def _normalize_qwen3_vl_text_encoder_state_dict(
  state_dict: dict[str, torch.Tensor],
  *,
  drop_visual: bool = False,
) -> dict[str, torch.Tensor]:
  """Adapt Comfy Qwen language-model keys to Transformers' wrapper names."""
  if any(key.startswith("language_model.") for key in state_dict):
    return state_dict

  language_prefixes = (
    "embed_tokens.",
    "layers.",
    "norm.",
    "rotary_emb.",
  )
  wrapper_prefixes = (
    "model.",
  )
  normalized: dict[str, torch.Tensor] = {}
  for key, value in state_dict.items():
    normalized_key = key
    for wrapper_prefix in wrapper_prefixes:
      if normalized_key.startswith(wrapper_prefix):
        normalized_key = normalized_key[len(wrapper_prefix) :]
        break

    if normalized_key.startswith("lm_head."):
      continue
    if drop_visual and normalized_key.startswith("visual."):
      continue
    if normalized_key.startswith(language_prefixes):
      normalized[f"language_model.{normalized_key}"] = value
    else:
      normalized[key] = value
  return normalized


def _is_comfy_qwen3_text_encoder_state_dict(
  state_dict: dict[str, torch.Tensor],
) -> bool:
  """True for Comfy's pruned Qwen3 text-encoder layout."""
  return any(key.startswith("model.embed_tokens.") for key in state_dict) and any(
    key.startswith("model.layers.") for key in state_dict
  )


class _ComfyQwen3TextEncoder(torch.nn.Module):
  """Wrapper exposing a Qwen3 text model through the pipeline's language_model API."""

  def __init__(self, language_model: torch.nn.Module) -> None:
    super().__init__()
    self.language_model = language_model
    self._ideogram_rope_mode = "qwen3_text"


def _make_comfy_qwen3_8b_config():
  from transformers import Qwen3Config

  config_kwargs = dict(
    vocab_size=151936,
    hidden_size=4096,
    intermediate_size=12288,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=40960,
    rms_norm_eps=1e-6,
    use_cache=False,
    tie_word_embeddings=False,
    attention_bias=False,
    use_sliding_window=False,
    pad_token_id=151643,
    eos_token_id=151645,
  )
  try:
    return Qwen3Config(
      **config_kwargs,
      rope_parameters={"rope_type": "default", "rope_theta": 5000000.0},
    )
  except TypeError:
    config = Qwen3Config(**config_kwargs)
    config.rope_theta = 5000000.0
    return config


def _materialize_qwen3_text_meta_init_buffers(model) -> None:
  """Materialize small fp32 rotary buffers skipped by meta construction."""
  text_rotary = model.language_model.rotary_emb
  if hasattr(text_rotary, "compute_default_rope_parameters"):
    inv_freq, text_rotary.attention_scaling = text_rotary.compute_default_rope_parameters(
      text_rotary.config, torch.device("cpu")
    )
    text_rotary.register_buffer("inv_freq", inv_freq, persistent=False)
    text_rotary.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)


def _build_comfy_qwen3_text_encoder():
  from transformers import Qwen3Model

  with torch.device("meta"):
    language_model = Qwen3Model(_make_comfy_qwen3_8b_config())
  model = _ComfyQwen3TextEncoder(language_model)
  _materialize_qwen3_text_meta_init_buffers(model)
  return model


def _load_subfolder_state_dict(
  repo_id: str, subfolder: str, basename: str
) -> dict[str, torch.Tensor]:
  """Download a component's weights, whether sharded (index) or a single file.

  ``basename`` is the safetensors stem (``model`` for transformers components,
  ``diffusion_pytorch_model`` for diffusers ones).
  """
  prefix = f"{subfolder}/" if subfolder else ""
  index_filename = f"{prefix}{basename}.safetensors.index.json"
  try:
    return _load_sharded_state_dict(repo_id, index_filename)
  except EntryNotFoundError:
    filename = f"{prefix}{basename}.safetensors"
    _load_log(f"Fetching single state dict file {filename}")
    t = time.perf_counter()
    single_path = hf_hub_download(repo_id=repo_id, filename=filename)
    _load_log(f"Reading single state dict file {filename}")
    state_dict = _load_safetensors_state_dict(single_path)
    _load_log(f"Loaded {filename} in {time.perf_counter() - t:.1f}s")
    return state_dict


def _load_state_dict_file(repo_id: str, filename: str) -> dict[str, torch.Tensor]:
  if filename.endswith(".index.json"):
    return _load_indexed_or_single_state_dict(repo_id, filename)
  _load_log(f"Fetching state dict file {filename}")
  t = time.perf_counter()
  path = hf_hub_download(repo_id=repo_id, filename=filename)
  _load_log(f"Reading state dict file {filename}")
  state_dict = _load_safetensors_state_dict(path)
  _load_log(f"Loaded {filename} in {time.perf_counter() - t:.1f}s")
  return state_dict


def _materialize_qwen3_vl_meta_init_buffers(model) -> None:
  """Materialize small fp32 rotary buffers skipped by meta construction."""
  text_rotary = model.language_model.rotary_emb
  inv_freq, text_rotary.attention_scaling = text_rotary.compute_default_rope_parameters(
    text_rotary.config, torch.device("cpu")
  )
  text_rotary.register_buffer("inv_freq", inv_freq, persistent=False)
  text_rotary.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

  vision_rotary = model.visual.rotary_pos_emb
  vision_inv_freq = 1.0 / (
    vision_rotary.theta
    ** (torch.arange(0, vision_rotary.dim, 2, dtype=torch.float32) / vision_rotary.dim)
  )
  vision_rotary.register_buffer("inv_freq", vision_inv_freq, persistent=False)


def _bnb4bit_quantization_config(
  quantization_config: object,
) -> dict[str, object] | None:
  """Return supported bnb 4-bit configs and reject other quantization formats."""
  if not isinstance(quantization_config, dict):
    return None
  if quantization_config.get("quant_method") != "bitsandbytes":
    return None
  if not (
    quantization_config.get("load_in_4bit")
    or quantization_config.get("_load_in_4bit")
  ):
    return None
  return dict(quantization_config)


def _load_bnb4bit_text_encoder(
  repo_id: str,
  device: torch.device,
  dtype: torch.dtype,
  *,
  text_encoder_subfolder: str,
  quantization_config: dict[str, object],
) -> torch.nn.Module:
  """Load bnb 4-bit Qwen on meta to avoid Transformers' slow bnb placement path."""
  t_total = time.perf_counter()
  _load_log(f"Loading bnb 4-bit text encoder config from {text_encoder_subfolder}")
  config = AutoConfig.from_pretrained(
    repo_id, subfolder=text_encoder_subfolder, trust_remote_code=True
  )
  state_dict = _load_subfolder_state_dict(repo_id, text_encoder_subfolder, "model")
  _load_log("Instantiating bnb 4-bit text encoder on meta device")
  with torch.device("meta"):
    model = AutoModel.from_config(config, trust_remote_code=True)
  _materialize_qwen3_vl_meta_init_buffers(model)
  _load_log("Swapping text encoder Linear layers to bnb 4-bit")
  with torch.device("meta"):
    swap_linears_to_bnb4bit(
      model,
      compute_dtype=dtype,
      quant_type=str(quantization_config.get("bnb_4bit_quant_type", "nf4")),
      compress_statistics=bool(
        quantization_config.get("bnb_4bit_use_double_quant", False)
      ),
    )
  _load_log("Assigning bnb 4-bit text encoder weights")
  load_bnb4bit_state_dict(model, state_dict, device=device, dtype=dtype)
  model.eval()
  _load_log(f"bnb 4-bit text encoder ready in {time.perf_counter() - t_total:.1f}s")
  return model


def _load_fp8_text_encoder(
  repo_id: str,
  device: torch.device,
  dtype: torch.dtype,
  *,
  text_encoder_subfolder: str,
):
  """Rebuild the text encoder from its config and load weight-only FP8 weights.

  transformers' ``from_pretrained`` can't read our float8 layout, so we
  instantiate the architecture with ``from_config`` (which also computes the
  non-persistent buffers such as rotary caches), swap the quantized Linears, and
  load the FP8 state dict with ``assign=True``.
  """
  t_total = time.perf_counter()
  _load_log(f"Loading FP8 text encoder config from {text_encoder_subfolder}")
  config = AutoConfig.from_pretrained(
    repo_id, subfolder=text_encoder_subfolder, trust_remote_code=True
  )
  state_dict = _load_subfolder_state_dict(repo_id, text_encoder_subfolder, "model")
  _load_log("Instantiating FP8 text encoder on meta device")
  with torch.device("meta"):
    model = AutoModel.from_config(config, trust_remote_code=True)
  _materialize_qwen3_vl_meta_init_buffers(model)
  _load_log("Swapping text encoder Linear layers to FP8")
  with torch.device("meta"):
    swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
  # assign=True so unquantized params take the loaded dtype and the computed
  # rotary buffers (absent from the checkpoint) survive; tied weights, if any,
  # surface as benign missing keys.
  _load_log("Moving FP8 text encoder weights to device")
  load_fp8_state_dict(
    model, state_dict, device=device, dtype=dtype, assign=True, strict=False
  )
  model.eval()
  _load_log(f"FP8 text encoder ready in {time.perf_counter() - t_total:.1f}s")
  return model


def _load_state_dict_text_encoder(
  *,
  weights_repo: str,
  weights_filename: str,
  config_repo: str,
  tokenizer_repo: str,
  device: torch.device,
  dtype: torch.dtype,
  tokenizer_subfolder: str | None = None,
  text_encoder_subfolder: str | None = None,
):
  """Load Qwen3-VL tokenizer/config separately from a single-file state dict."""
  t_total = time.perf_counter()
  tokenizer_kwargs = {"subfolder": tokenizer_subfolder} if tokenizer_subfolder else {}
  config_kwargs = {"subfolder": text_encoder_subfolder} if text_encoder_subfolder else {}

  _load_log(f"Loading tokenizer from {tokenizer_repo}/{tokenizer_subfolder or ''}")
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, **tokenizer_kwargs)

  _load_log(f"Loading text encoder config from {config_repo}/{text_encoder_subfolder or ''}")
  config = AutoConfig.from_pretrained(
    config_repo, trust_remote_code=True, **config_kwargs
  )

  state_dict = _load_state_dict_file(weights_repo, weights_filename)
  state_dict = _normalize_state_dict_prefix(
    state_dict,
    (
      "language_model.embed_tokens.weight",
      "model.embed_tokens.weight",
      "visual.patch_embed.proj.weight",
    ),
  )
  use_comfy_qwen3_text_encoder = _is_comfy_qwen3_text_encoder_state_dict(state_dict)
  if use_comfy_qwen3_text_encoder:
    _load_log("Dropping Qwen3-VL visual tower weights for text-only Ideogram conditioning")
  state_dict = _normalize_qwen3_vl_text_encoder_state_dict(
    state_dict, drop_visual=use_comfy_qwen3_text_encoder
  )

  if use_comfy_qwen3_text_encoder:
    _load_log("Instantiating Comfy Qwen3 text encoder on meta device")
    model = _build_comfy_qwen3_text_encoder()
  else:
    _load_log("Instantiating text encoder on meta device")
    with torch.device("meta"):
      model = AutoModel.from_config(config, trust_remote_code=True)
    _materialize_qwen3_vl_meta_init_buffers(model)

  if is_comfy_quant_state_dict(state_dict):
    if not is_nvfp4_state_dict(state_dict):
      _load_log("Text encoder has Comfy FP8 quantized weights")
    else:
      _load_log("Text encoder has Comfy NVFP4 quantized weights")
    _load_log("Swapping text encoder Linear layers to Comfy quant")
    with torch.device("meta"):
      swap_linears_to_comfy_quant(model, state_dict, compute_dtype=dtype)
    _load_log("Assigning Comfy quant text encoder weights")
    load_comfy_quant_state_dict(
      model, state_dict, device=device, dtype=dtype, assign=True, strict=False
    )
  elif is_fp8_state_dict(state_dict):
    _load_log("Swapping text encoder Linear layers to FP8")
    with torch.device("meta"):
      swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
    load_fp8_state_dict(
      model, state_dict, device=device, dtype=dtype, assign=True, strict=False
    )
  else:
    prepared = {
      k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
      for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(prepared, strict=False, assign=True)
    if unexpected:
      raise RuntimeError(f"unexpected keys after text encoder load: {unexpected[:10]}")
    if missing:
      warnings.warn(f"missing keys after text encoder load: {missing[:10]}", stacklevel=2)
    model.to(device)

  model.eval()
  _load_log(f"State-dict text encoder ready in {time.perf_counter() - t_total:.1f}s")
  return tokenizer, model


def _load_qwen3_vl(
  repo_id: str,
  device: torch.device,
  dtype: torch.dtype,
  *,
  tokenizer_subfolder: str | None = None,
  text_encoder_subfolder: str | None = None,
):
  """Load the Qwen3-VL tokenizer + model, optionally from named subfolders of ``repo_id``.

  When the weights are published in diffusers layout the tokenizer lives at ``tokenizer/``
  and the model at ``text_encoder/`` within the same repo as the transformer weights, so
  there is no need to fetch them from a separate upstream repo.

  """
  tokenizer_kwargs = {"subfolder": tokenizer_subfolder} if tokenizer_subfolder else {}
  model_kwargs = {"subfolder": text_encoder_subfolder} if text_encoder_subfolder else {}
  t_total = time.perf_counter()
  _load_log(f"Loading tokenizer from {tokenizer_subfolder or repo_id}")
  tokenizer = AutoTokenizer.from_pretrained(repo_id, **tokenizer_kwargs)

  _load_log(f"Fetching text encoder config from {text_encoder_subfolder or repo_id}")
  cfg_path = hf_hub_download(
    repo_id=repo_id,
    filename=f"{text_encoder_subfolder}/config.json"
    if text_encoder_subfolder
    else "config.json",
  )
  with open(cfg_path) as f:
    cfg_data = json.load(f)
  quantization_config = cfg_data.get("quantization_config")
  bnb4bit_config = _bnb4bit_quantization_config(quantization_config)
  is_fp8 = bool(cfg_data.get(FP8_TEXT_ENCODER_CONFIG_FLAG, False))

  if is_fp8:
    model = _load_fp8_text_encoder(
      repo_id,
      device,
      dtype,
      text_encoder_subfolder=text_encoder_subfolder or "",
    )
  elif bnb4bit_config is not None:
    model = _load_bnb4bit_text_encoder(
      repo_id,
      device,
      dtype,
      text_encoder_subfolder=text_encoder_subfolder or "",
      quantization_config=bnb4bit_config,
    )
  elif quantization_config is not None:
    _load_log("Loading quantized text encoder with transformers")
    model = AutoModel.from_pretrained(
      repo_id,
      torch_dtype=dtype,
      trust_remote_code=True,
      device_map={"": device},
      **model_kwargs,
    )
    model.eval()
  else:
    _load_log("Loading text encoder with transformers")
    model = AutoModel.from_pretrained(
      repo_id, torch_dtype=dtype, trust_remote_code=True, **model_kwargs
    )
    model.to(device)
    model.eval()
  _load_log(f"Qwen3-VL tokenizer/text encoder ready in {time.perf_counter() - t_total:.1f}s")
  return tokenizer, model


def build_meta_transformer(
  transformer_config: "Ideogram4Config",
) -> "Ideogram4Transformer":
  """Construct on meta to avoid initializing full-size weights before assign-loading."""
  with torch.device("meta"):
    model = Ideogram4Transformer(transformer_config)
  head_dim = transformer_config.emb_dim // transformer_config.num_heads
  inv_freq = 1.0 / (
    transformer_config.rope_theta
    ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
  )
  model.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
  return model


def _build_transformer(
  transformer_config: "Ideogram4Config",
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
  *,
  name: str = "transformer",
) -> "Ideogram4Transformer":
  t = time.perf_counter()
  _load_log(f"Building {name}")
  state_dict = _normalize_state_dict_prefix(
    state_dict,
    (
      "input_proj.weight",
      "layers.0.attention.qkv.weight",
    ),
  )
  if is_comfy_quant_state_dict(state_dict):
    if not is_nvfp4_state_dict(state_dict):
      raise RuntimeError(f"{name}: Comfy quantized state dict is not NVFP4")
    _load_log(f"{name}: detected Comfy NVFP4 state dict")
    model = build_meta_transformer(transformer_config)
    with torch.device("meta"):
      swap_linears_to_comfy_quant(model, state_dict, compute_dtype=dtype)
    load_comfy_quant_state_dict(
      model, state_dict, device=device, dtype=dtype, assign=True
    )
  elif is_bnb4bit_state_dict(state_dict):
    _load_log(f"{name}: detected bitsandbytes 4-bit state dict")
    if device.type != "cuda":
      raise ValueError(f"bnb 4-bit weights require a CUDA device, got device={device}")
    model = build_meta_transformer(transformer_config)
    with torch.device("meta"):
      swap_linears_to_bnb4bit(model, compute_dtype=dtype)
    load_bnb4bit_state_dict(model, state_dict, device=device, dtype=dtype)
  elif is_fp8_state_dict(state_dict):
    _load_log(f"{name}: detected FP8 state dict")
    model = build_meta_transformer(transformer_config)
    with torch.device("meta"):
      swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
    load_fp8_state_dict(model, state_dict, device=device, dtype=dtype, assign=True)
  else:
    _load_log(f"{name}: detected regular state dict")
    model = Ideogram4Transformer(transformer_config)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
  model.eval()
  _load_log(f"{name} ready in {time.perf_counter() - t:.1f}s")
  return model


def _load_autoencoder(weights_path: str, device: torch.device, dtype: torch.dtype):
  t = time.perf_counter()
  _load_log("Loading VAE/autoencoder")
  ae = AutoEncoder(AutoEncoderParams())
  raw_state_dict = _load_safetensors_state_dict(weights_path)
  errors: list[str] = []
  candidates: list[dict[str, torch.Tensor]] = []
  try:
    candidates.append(convert_diffusers_state_dict(raw_state_dict))
  except Exception as exc:
    errors.append(str(exc))
  candidates.append(raw_state_dict)

  for state_dict in candidates:
    try:
      ae.load_state_dict(state_dict)
      break
    except RuntimeError as exc:
      errors.append(str(exc))
  else:
    raise RuntimeError("Could not load VAE/autoencoder state dict:\n" + "\n".join(errors[:2]))
  ae.to(device=device, dtype=dtype)
  ae.eval()
  _load_log(f"VAE/autoencoder ready in {time.perf_counter() - t:.1f}s")
  return ae


def _load_sharded_state_dict(
  repo_id: str, index_filename: str
) -> dict[str, torch.Tensor]:
  """Download a sharded safetensors checkpoint and merge it into one state dict.

  ``index_filename`` is the path of the safetensors index file inside the repo
  (e.g. ``conditional_model/model.safetensors.index.json``). Shard filenames in
  the index are interpreted relative to that index's directory, matching the
  layout written by ``huggingface_hub.save_torch_state_dict``.
  """
  _load_log(f"Fetching shard index {index_filename}")
  index_path = hf_hub_download(repo_id=repo_id, filename=index_filename)
  with open(index_path) as f:
    index = json.load(f)
  weight_map: dict[str, str] = index["weight_map"]
  shard_dir = _posix_dirname(index_filename)
  shard_filenames = sorted(set(weight_map.values()))

  state_dict: dict[str, torch.Tensor] = {}
  _load_log(f"Loading {len(shard_filenames)} shard(s) for {index_filename}")
  for shard_num, shard in enumerate(shard_filenames, start=1):
    shard_repo_path = _posix_join(shard_dir, shard) if shard_dir else shard
    t = time.perf_counter()
    _load_log(f"[{shard_num}/{len(shard_filenames)}] Fetching {shard_repo_path}")
    shard_path = hf_hub_download(repo_id=repo_id, filename=shard_repo_path)
    _load_log(f"[{shard_num}/{len(shard_filenames)}] Reading {shard_repo_path}")
    state_dict.update(_load_safetensors_state_dict(shard_path))
    _load_log(
      f"[{shard_num}/{len(shard_filenames)}] Loaded {shard_repo_path} "
      f"in {time.perf_counter() - t:.1f}s"
    )
  return state_dict


def _load_indexed_or_single_state_dict(
  repo_id: str, index_filename: str
) -> dict[str, torch.Tensor]:
  """Load a component whether published as a sharded index or a single file.

  Some repos publish each component as a single ``.safetensors`` file rather
  than a sharded checkpoint with an ``.index.json``. Try the index first and
  fall back to the single file (the index filename with ``.index.json``
  dropped) when it isn't present.
  """
  try:
    return _load_sharded_state_dict(repo_id, index_filename)
  except EntryNotFoundError:
    single_filename = index_filename.removesuffix(".index.json")
    _load_log(f"Fetching single state dict file {single_filename}")
    t = time.perf_counter()
    single_path = hf_hub_download(repo_id=repo_id, filename=single_filename)
    _load_log(f"Reading single state dict file {single_filename}")
    state_dict = _load_safetensors_state_dict(single_path)
    _load_log(f"Loaded {single_filename} in {time.perf_counter() - t:.1f}s")
    return state_dict


@dataclass
class Ideogram4PipelineConfig:
  weights_repo: str = "ideogram-ai/ideogram-4-nf4"
  conditional_weights_repo: Optional[str] = None
  unconditional_weights_repo: Optional[str] = None
  autoencoder_weights_repo: Optional[str] = None
  conditional_index_filename: str = (
    "transformer/diffusion_pytorch_model.safetensors.index.json"
  )
  unconditional_index_filename: str = (
    "unconditional_transformer/diffusion_pytorch_model.safetensors.index.json"
  )
  autoencoder_filename: str = "vae/diffusion_pytorch_model.safetensors"
  tokenizer_repo: Optional[str] = None
  text_encoder_config_repo: Optional[str] = None
  text_encoder_weights_repo: Optional[str] = None
  text_encoder_weights_filename: Optional[str] = None
  text_encoder_subfolder: str = "text_encoder"
  tokenizer_subfolder: str = "tokenizer"
  patch_size: int = 2
  ae_scale_factor: int = 8
  max_text_tokens: int = 8192


class Ideogram4Pipeline:
  """Ideogram 4 text-to-image pipeline."""

  def __init__(
    self,
    conditional_transformer: Ideogram4Transformer,
    unconditional_transformer: Ideogram4Transformer,
    text_encoder,
    text_tokenizer,
    autoencoder,
    config: Ideogram4PipelineConfig,
    device: torch.device,
    dtype: torch.dtype,
  ) -> None:
    self.conditional_transformer = conditional_transformer
    self.unconditional_transformer = unconditional_transformer
    self.text_encoder = text_encoder
    self.text_tokenizer = text_tokenizer
    self.autoencoder = autoencoder
    self.config = config
    self.device = device
    self.dtype = dtype
    self.caption_verifier = CaptionVerifier()

    shift, scale = get_latent_norm()
    self.latent_shift = shift.to(device)
    self.latent_scale = scale.to(device)

  @classmethod
  def from_pretrained(
    cls,
    *,
    config: Optional[Ideogram4PipelineConfig] = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    transformer_config: Optional[Ideogram4Config] = None,
  ) -> "Ideogram4Pipeline":
    config = config or Ideogram4PipelineConfig()
    transformer_config = transformer_config or Ideogram4Config()
    device = torch.device(device)

    t_total = time.perf_counter()
    _load_log(
      f"Starting Ideogram4Pipeline.from_pretrained repo={config.weights_repo} "
      f"device={device} dtype={dtype}"
    )
    conditional_repo = config.conditional_weights_repo or config.weights_repo
    unconditional_repo = config.unconditional_weights_repo or config.weights_repo
    autoencoder_repo = config.autoencoder_weights_repo or config.weights_repo

    t = time.perf_counter()
    _load_log(f"Loading conditional transformer state dict from {conditional_repo}")
    conditional_state_dict = _load_indexed_or_single_state_dict(
      conditional_repo, config.conditional_index_filename
    )
    _load_log(f"Conditional transformer state dict loaded in {time.perf_counter() - t:.1f}s")
    conditional_transformer = _build_transformer(
      transformer_config, conditional_state_dict, device, dtype, name="conditional transformer"
    )
    del conditional_state_dict

    t = time.perf_counter()
    _load_log(f"Loading unconditional transformer state dict from {unconditional_repo}")
    unconditional_state_dict = _load_indexed_or_single_state_dict(
      unconditional_repo, config.unconditional_index_filename
    )
    _load_log(f"Unconditional transformer state dict loaded in {time.perf_counter() - t:.1f}s")
    unconditional_transformer = _build_transformer(
      transformer_config, unconditional_state_dict, device, dtype, name="unconditional transformer"
    )
    del unconditional_state_dict

    t = time.perf_counter()
    _load_log(f"Fetching autoencoder weights {config.autoencoder_filename} from {autoencoder_repo}")
    autoencoder_weights = hf_hub_download(
      repo_id=autoencoder_repo, filename=config.autoencoder_filename
    )
    _load_log(f"Autoencoder weights ready in {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    _load_log("Loading text tokenizer/text encoder")
    if config.text_encoder_weights_filename:
      text_tokenizer, text_encoder = _load_state_dict_text_encoder(
        weights_repo=config.text_encoder_weights_repo or config.weights_repo,
        weights_filename=config.text_encoder_weights_filename,
        config_repo=config.text_encoder_config_repo or config.weights_repo,
        tokenizer_repo=config.tokenizer_repo or config.weights_repo,
        device=device,
        dtype=dtype,
        tokenizer_subfolder=config.tokenizer_subfolder,
        text_encoder_subfolder=config.text_encoder_subfolder,
      )
    else:
      text_tokenizer, text_encoder = _load_qwen3_vl(
        config.weights_repo,
        device,
        dtype,
        tokenizer_subfolder=config.tokenizer_subfolder,
        text_encoder_subfolder=config.text_encoder_subfolder,
      )
    _load_log(f"Text tokenizer/text encoder loaded in {time.perf_counter() - t:.1f}s")
    autoencoder = _load_autoencoder(autoencoder_weights, device, dtype)

    pipeline = cls(
      conditional_transformer=conditional_transformer,
      unconditional_transformer=unconditional_transformer,
      text_encoder=text_encoder,
      text_tokenizer=text_tokenizer,
      autoencoder=autoencoder,
      config=config,
      device=device,
      dtype=dtype,
    )
    _load_log(f"Ideogram4Pipeline ready in {time.perf_counter() - t_total:.1f}s")
    return pipeline

  def _tokenize(self, prompt: str) -> tuple[torch.Tensor, int]:
    """Build chat-formatted token ids for a single prompt."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = self.text_tokenizer.apply_chat_template(
      messages, add_generation_prompt=True, tokenize=False
    )
    encoded = self.text_tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    num_text_tokens = int(token_ids.shape[0])
    if num_text_tokens > self.config.max_text_tokens:
      raise ValueError(
        f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={self.config.max_text_tokens}"
      )
    return token_ids, num_text_tokens

  def _build_inputs(
    self,
    prompts: list[str],
    height: int,
    width: int,
  ) -> dict[str, object]:
    """Build the packed sequence (text tokens + image tokens) for one batch."""
    tokenized = [self._tokenize(p) for p in prompts]
    batch_size = len(prompts)

    patch = self.config.patch_size * self.config.ae_scale_factor
    if height % patch != 0 or width % patch != 0:
      raise ValueError(
        f"height/width must be divisible by patch_size*ae_scale_factor={patch}"
      )
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w

    max_text_tokens = max(num_text for _, num_text in tokenized)
    total_seq_len = max_text_tokens + num_image_tokens
    no_padding = all(num_text == max_text_tokens for _, num_text in tokenized)

    # Image position ids (t=0, h, w) offset to keep them disjoint from text positions.
    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
    text_position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)

    segment_ids = torch.full(
      (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long
    )
    indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

    for b, (toks, num_text) in enumerate(tokenized):
      pad_len = max_text_tokens - num_text
      total_unpadded = num_text + num_image_tokens

      # Layout: [pad_len zeros] [text tokens] [image tokens]
      offset = pad_len
      token_ids[b, offset : offset + num_text] = toks
      # Image token slots stay at 0.

      text_pos = torch.arange(num_text)
      text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
      text_position_ids[b, offset : offset + num_text] = text_pos_3d
      position_ids[b, offset : offset + num_text] = text_pos_3d
      position_ids[b, offset + num_text :] = image_pos

      indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
      indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

      # Segment id 1 for the (text+image) sample, padding stays at 0.
      segment_ids[b, offset : offset + total_unpadded] = 1

    segment_ids = segment_ids.to(self.device)

    return {
      "token_ids": token_ids.to(self.device),
      "text_position_ids": text_position_ids.to(self.device),
      "position_ids": position_ids.to(self.device),
      "segment_ids": segment_ids,
      "attention_segment_ids": None if no_padding else segment_ids,
      "indicator": indicator.to(self.device),
      "num_image_tokens": num_image_tokens,  # type: ignore[dict-item]
      "grid_h": grid_h,  # type: ignore[dict-item]
      "grid_w": grid_w,  # type: ignore[dict-item]
      "max_text_tokens": max_text_tokens,  # type: ignore[dict-item]
    }

  def _get_qwen3_vl_embeddings(
    self,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_2d: torch.Tensor,
  ) -> list[torch.Tensor]:
    language_model = self.text_encoder.language_model

    inputs_embeds = language_model.embed_tokens(token_ids)

    text_position_ids = pos_2d

    causal_mask = create_causal_mask(
      config=language_model.config,
      inputs_embeds=inputs_embeds,
      attention_mask=attention_mask,
      past_key_values=None,
      position_ids=text_position_ids,
    )
    if getattr(self.text_encoder, "_ideogram_rope_mode", None) == "qwen3_text":
      position_embeddings = language_model.rotary_emb(inputs_embeds, text_position_ids)
    else:
      position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
      mrope_position_ids = position_ids_4d[1:]
      position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
      hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask,
        position_ids=text_position_ids,
        past_key_values=None,
        position_embeddings=position_embeddings,
      )
      if layer_idx in tap_set:
        captured[layer_idx] = hidden_states

    return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]

  def _encode_text(
    self,
    token_ids: torch.Tensor,
    text_position_ids: torch.Tensor,
    indicator: torch.Tensor,
  ) -> torch.Tensor:
    """Run Qwen3-VL and stack hidden states from the activation layers.

    Returns a (B, L, hidden_size * num_layers) float32 tensor.
    """
    batch_size, seq_len = token_ids.shape

    # Real text positions are exactly the LLM_TOKEN_INDICATOR positions.
    attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)

    pos_2d = text_position_ids[..., 0].contiguous()

    with torch.no_grad():
      selected = self._get_qwen3_vl_embeddings(token_ids, attention_mask, pos_2d)
    stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))
    stacked = stacked.reshape(batch_size, seq_len, -1)

    # Zero out non-LLM positions (left padding) so the transformer only sees real
    # text features at LLM_TOKEN_INDICATOR positions.
    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    stacked = stacked * text_mask
    return stacked.to(torch.float32)

  def _verify_prompts(
    self, prompts: list[str], *, raise_on_issues: bool = True
  ) -> None:
    """Run each prompt through the caption verifier.

    Raises ``ValueError`` if any prompt has issues. When ``raise_on_issues``
    is False, issues are emitted as warnings instead.
    """
    messages: list[str] = []
    for i, prompt in enumerate(prompts):
      issues = self.caption_verifier.verify_raw(prompt)
      if not issues:
        continue
      messages.append(f"caption verifier flagged prompt[{i}]:\n" + "\n".join(issues))
    if not messages:
      return
    combined = "\n".join(messages)
    if raise_on_issues:
      raise ValueError(combined)
    warnings.warn(combined, stacklevel=2)

  @torch.no_grad()
  def __call__(
    self,
    prompts: str | list[str],
    *,
    height: int = 1024,
    width: int = 1024,
    num_steps: int = 128,
    guidance_scale: float = 7.0,
    guidance_schedule: Optional[Sequence[float] | torch.Tensor] = None,
    mu: float = 0.5,
    std: float = 1.0,
    seed: Optional[int] = None,
    schedule: Optional[LogitNormalSchedule] = None,
    raise_on_caption_issues: bool = True,
    output_type: str = "pil",
    callback_on_step_end: Optional[Callable[[int, int], None]] = None,
  ) -> list[Image.Image] | dict[str, torch.Tensor]:
    """Generate images for the given prompts."""
    if isinstance(prompts, str):
      prompts = [prompts]

    self._verify_prompts(prompts, raise_on_issues=raise_on_caption_issues)

    schedule = schedule or get_schedule_for_resolution(
      (height, width), known_mean=mu, std=std
    )
    step_intervals = make_step_intervals(num_steps).to(self.device)

    if guidance_schedule is not None:
      gw_per_step = torch.as_tensor(
        guidance_schedule, dtype=torch.float32, device=self.device
      )
      if gw_per_step.shape != (num_steps,):
        raise ValueError(
          f"guidance_schedule must have shape ({num_steps},), "
          f"got {tuple(gw_per_step.shape)}"
        )
    else:
      gw_per_step = torch.full(
        (num_steps,), float(guidance_scale), dtype=torch.float32, device=self.device
      )
    skip_uncond_per_step = [
      abs(float(gw) - 1.0) <= 1e-6 for gw in gw_per_step.detach().cpu()
    ]

    inputs = self._build_inputs(prompts, height=height, width=width)
    batch_size = len(prompts)
    num_image_tokens = inputs["num_image_tokens"]
    grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]
    max_text_tokens = inputs["max_text_tokens"]
    latent_dim = self.conditional_transformer.config.in_channels

    llm_features = self._encode_text(
      inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
    )

    # Negative branch is image-only (asymmetric CFG) with zeroed conditioning.
    neg_position_ids = inputs["position_ids"][:, max_text_tokens:]
    neg_indicator = inputs["indicator"][:, max_text_tokens:]
    neg_llm_features = torch.zeros(  # type: ignore[call-overload]
      batch_size,
      num_image_tokens,
      llm_features.shape[-1],
      dtype=llm_features.dtype,
      device=self.device,
    )

    generator = torch.Generator(device=self.device)
    if seed is not None:
      generator.manual_seed(seed)
    z = torch.randn(  # type: ignore[call-overload]
      batch_size,
      num_image_tokens,
      latent_dim,
      dtype=torch.float32,
      device=self.device,
      generator=generator,
    )

    text_z_padding = torch.zeros(  # type: ignore[call-overload]
      batch_size,
      max_text_tokens,
      latent_dim,
      dtype=torch.float32,
      device=self.device,
    )

    schedule_values = schedule(step_intervals)
    for i in range(num_steps - 1, -1, -1):
      t = schedule_values[i + 1].expand(batch_size)

      pos_z = torch.cat([text_z_padding, z], dim=1)
      pos_out = self.conditional_transformer(
        llm_features=llm_features,
        x=pos_z,
        t=t,
        position_ids=inputs["position_ids"],
        segment_ids=inputs["attention_segment_ids"],
        indicator=inputs["indicator"],
      )
      pos_v = pos_out[:, max_text_tokens:]

      gw_i = gw_per_step[i]
      if skip_uncond_per_step[i]:
        v = pos_v
      else:
        neg_v = self.unconditional_transformer(
          llm_features=neg_llm_features,
          x=z,
          t=t,
          position_ids=neg_position_ids,
          segment_ids=None,
          indicator=neg_indicator,
        )
        v = gw_i * pos_v + (1.0 - gw_i) * neg_v
      delta = schedule_values[i] - schedule_values[i + 1]
      z = z + v * delta
      if callback_on_step_end is not None:
        callback_on_step_end(num_steps - i, num_steps)

    if output_type == "pil":
      return self._decode(z, grid_h=grid_h, grid_w=grid_w)  # type: ignore[arg-type]
    if output_type == "latent":
      latents = self._unpack_latents(z, grid_h=grid_h, grid_w=grid_w)
      return {
        "latents": latents,
        "decoded": self._decode_latent_tensor(latents),
      }
    raise ValueError(f"Unsupported output_type: {output_type!r}")

  def _unpack_latents(self, z: torch.Tensor, *, grid_h: int, grid_w: int) -> torch.Tensor:
    """Denormalize and unpatch model latents for the Flux2 VAE decoder."""
    batch_size = z.shape[0]
    patch = self.config.patch_size

    z = z * self.latent_scale + self.latent_shift

    ae_channels = z.shape[-1] // (patch * patch)
    z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    z = z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)

    return z.to(self.dtype)

  def _decode_latent_tensor(self, z: torch.Tensor) -> torch.Tensor:
    """Run the autoencoder decoder and return RGB tensors in [-1, 1]."""
    return self.autoencoder.decoder(z.to(self.dtype)).float().clamp(-1.0, 1.0)

  def _decode(self, z: torch.Tensor, *, grid_h: int, grid_w: int) -> list[Image.Image]:
    """Unpatch and run the autoencoder decoder."""
    z = self._unpack_latents(z, grid_h=grid_h, grid_w=grid_w)
    decoded = self._decode_latent_tensor(z)

    decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
    decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(arr) for arr in decoded]
