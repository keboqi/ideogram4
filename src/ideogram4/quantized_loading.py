from __future__ import annotations

import json
import warnings

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F


_BNB_SIBLING_SUFFIXES = (
  ".absmax",
  ".quant_map",
  ".nested_absmax",
  ".nested_quant_map",
)

# Largest magnitude representable by the e4m3 float8 format. Per-row weight
# scales map each row's max abs value onto this so we use the full range.
FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"
# Marker written into the text encoder's config.json so the loader knows to take
# the custom weight-only FP8 path instead of transformers' from_pretrained.
FP8_TEXT_ENCODER_CONFIG_FLAG = "ideogram_fp8_weight_only"
COMFY_QUANT_SUFFIX = ".comfy_quant"
COMFY_FP8_FORMAT = "float8_e4m3fn"
COMFY_FP8_LAYOUT = "TensorCoreFP8Layout"
COMFY_NVFP4_FORMAT = "nvfp4"
COMFY_NVFP4_LAYOUT = "TensorCoreNVFP4Layout"


def _decode_comfy_quant_config(tensor: torch.Tensor | None) -> dict[str, object] | None:
  """Decode ComfyUI's per-layer quantization JSON tensor."""
  if tensor is None or not torch.is_tensor(tensor):
    return None
  try:
    data = bytes(tensor.detach().cpu().to(torch.uint8).flatten().tolist()).rstrip(b"\x00")
    if not data:
      return None
    config = json.loads(data.decode("utf-8"))
    return config if isinstance(config, dict) else None
  except Exception:
    return None


def _load_comfy_kitchen():
  try:
    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class
  except ImportError as exc:
    raise RuntimeError(
      "NVFP4 checkpoints require comfy-kitchen. Install it with "
      "`pip install comfy-kitchen[cublas]` for Blackwell CUDA acceleration "
      "or `pip install comfy-kitchen` for the slower eager fallback."
    ) from exc
  return QuantizedTensor, get_layout_class


def is_comfy_quant_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  return any(k.endswith(COMFY_QUANT_SUFFIX) for k in state_dict)


def is_nvfp4_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  for key, tensor in state_dict.items():
    if not key.endswith(COMFY_QUANT_SUFFIX):
      continue
    config = _decode_comfy_quant_config(tensor)
    if config and config.get("format") == COMFY_NVFP4_FORMAT:
      return True
  return False


def is_bnb4bit_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if any key looks like a bnb 4-bit quant_state sibling."""
  return any(".quant_state.bitsandbytes__" in k for k in state_dict)


def swap_linears_to_bnb4bit(
  module: nn.Module,
  compute_dtype: torch.dtype,
  *,
  quant_type: str = "nf4",
  compress_statistics: bool = False,
) -> None:
  for name, child in list(module.named_children()):
    if isinstance(child, nn.Linear):
      new_linear = bnb.nn.Linear4bit(
        child.in_features,
        child.out_features,
        bias=child.bias is not None,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
      )
      setattr(module, name, new_linear)
    else:
      swap_linears_to_bnb4bit(
        child,
        compute_dtype,
        quant_type=quant_type,
        compress_statistics=compress_statistics,
      )


def load_bnb4bit_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
) -> None:
  """Load a bnb 4-bit checkpoint by assigning prepared tensors into the model."""
  consumed: set[str] = set()
  for full_name, tensor in state_dict.items():
    if ".quant_state." in full_name or full_name.endswith(_BNB_SIBLING_SUFFIXES):
      continue
    parent_path, _, param_name = full_name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    current = parent._parameters.get(param_name)
    if not isinstance(current, bnb.nn.Params4bit):
      continue
    prefix = full_name + "."
    quantized_stats = {
      name: stat for name, stat in state_dict.items() if name.startswith(prefix)
    }
    # bnb's from_prequantized may mutate the stats dict, so snapshot names first.
    consumed.add(full_name)
    consumed.update(quantized_stats.keys())
    parent._parameters[param_name] = bnb.nn.Params4bit.from_prequantized(
      data=tensor,
      quantized_stats=quantized_stats,
      requires_grad=False,
      device=device,
    )

  prepared_remaining = {}
  for name, tensor in state_dict.items():
    if name in consumed:
      continue
    if tensor.is_floating_point():
      prepared_remaining[name] = tensor.to(device=device, dtype=dtype)
    else:
      prepared_remaining[name] = tensor.to(device=device)

  missing, unexpected = model.load_state_dict(
    prepared_remaining, strict=False, assign=True
  )
  # Quantized weights are loaded via from_prequantized above, so they appear in
  # `missing` from load_state_dict's perspective — filter those out.
  real_missing = [m for m in missing if m not in consumed]
  if real_missing:
    raise RuntimeError(f"missing keys after quantized load: {real_missing[:10]}")
  if unexpected:
    raise RuntimeError(f"unexpected keys after quantized load: {unexpected[:10]}")

  for name, buffer in list(model.named_buffers()):
    parent_path, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    persistent = leaf not in parent._non_persistent_buffers_set
    if persistent and buffer.is_floating_point():
      moved = buffer.to(device=device, dtype=dtype)
    else:
      moved = buffer.to(device=device)
    if moved is not buffer:
      parent.register_buffer(leaf, moved, persistent=persistent)


# ---------------------------------------------------------------------------
# Weight-only FP8 (e4m3)
#
# Activations stay in the compute dtype (e.g. bfloat16); only Linear weights are
# stored as float8 with a per-output-channel (per-row) float32 scale. At forward
# time the weight is dequantized back to the compute dtype and a normal bf16
# matmul runs, so this needs no FP8 tensor-core hardware and works on any device
# that can store float8 (CPU included). The win is ~2x smaller Linear weights.
# ---------------------------------------------------------------------------


def quantize_weight_to_fp8(
  weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize a 2-D Linear weight to e4m3 float8 with per-row scales.

  Returns ``(weight_fp8, scale)`` where ``weight_fp8`` has shape ``(out, in)``
  in ``float8_e4m3fn`` and ``scale`` has shape ``(out,)`` in float32 such that
  ``weight ≈ weight_fp8.to(dtype) * scale[:, None]``.
  """
  w = weight.detach().to(torch.float32)
  amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
  scale = amax / FP8_E4M3_MAX
  q = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
  return q, scale.squeeze(1).to(torch.float32)


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if the checkpoint carries weight-only FP8 Linear weights."""
  return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
    v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
  )


class Fp8Linear(nn.Module):
  """Linear layer holding an e4m3 float8 weight + per-row float32 scale.

  The weight and scale are registered as buffers (not parameters) so they load
  via ``load_state_dict`` and are excluded from optimizer/grad machinery. The
  dequantized matmul runs in ``compute_dtype``.
  """

  weight: torch.Tensor
  weight_scale: torch.Tensor
  bias: torch.Tensor | None

  def __init__(
    self,
    in_features: int,
    out_features: int,
    bias: bool,
    compute_dtype: torch.dtype,
  ) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.compute_dtype = compute_dtype
    self.register_buffer(
      "weight",
      torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
    )
    self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
    if bias:
      self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
    else:
      self.bias = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    scale = self.weight_scale.to(x.dtype)
    if scale.ndim == 0:
      w = self.weight.to(x.dtype) * scale
    else:
      w = self.weight.to(x.dtype) * scale.unsqueeze(1)
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, w, bias)


def swap_linears_to_fp8(
  module: nn.Module,
  state_dict: dict[str, torch.Tensor],
  compute_dtype: torch.dtype,
  *,
  prefix: str = "",
) -> None:
  """Replace each ``nn.Linear`` that has a saved FP8 scale with an ``Fp8Linear``.

  Gating on the presence of ``<name>.weight_scale`` means only layers that were
  actually quantized at save time are swapped; everything else loads normally in
  the compute dtype.
  """
  for name, child in list(module.named_children()):
    child_prefix = f"{prefix}{name}"
    if (
      isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict
    ):
      setattr(
        module,
        name,
        Fp8Linear(
          child.in_features,
          child.out_features,
          bias=child.bias is not None,
          compute_dtype=compute_dtype,
        ),
      )
    else:
      swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


def load_fp8_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
  *,
  assign: bool = False,
  strict: bool = True,
) -> None:
  """Load a weight-only FP8 checkpoint into ``model``.

  ``model`` must already have its FP8 Linear layers swapped in (see
  ``swap_linears_to_fp8``). FP8 weights are kept as float8, scales stay float32,
  and every other floating tensor is cast to ``dtype``.

  ``assign=True`` replaces the module's tensors with the prepared ones rather than
  copying into them. Use it when the model was built with ``from_config`` so the
  non-quantized params take the loaded dtype directly and computed non-persistent
  buffers (e.g. rotary caches) are left untouched. With ``assign=False`` (default),
  the caller must have already put the unquantized params in ``dtype``.

  ``strict=False`` downgrades missing keys to a warning (e.g. tied weights that a
  ``transformers`` model resolves itself); unexpected keys always raise.
  """
  prepared: dict[str, torch.Tensor] = {}
  for k, v in state_dict.items():
    if v.dtype == FP8_WEIGHT_DTYPE:
      prepared[k] = v.to(device=device)
    elif k.endswith(FP8_SCALE_SUFFIX):
      prepared[k] = v.to(device=device, dtype=torch.float32)
    elif v.is_floating_point():
      prepared[k] = v.to(device=device, dtype=dtype)
    else:
      prepared[k] = v.to(device=device)

  missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
  if unexpected:
    raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
  if missing:
    if strict:
      raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
    warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)

  model.to(device)


class ComfyQuantLinear(nn.Module):
  """Linear layer backed by a ComfyUI QuantizedTensor weight."""

  weight: torch.Tensor
  bias: torch.Tensor | None

  def __init__(
    self,
    in_features: int,
    out_features: int,
    bias: bool,
    compute_dtype: torch.dtype,
    layout_type: str,
  ) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.compute_dtype = compute_dtype
    self.layout_type = layout_type
    self._orig_shape = (out_features, in_features)
    self._fallback_warned = False
    self.full_precision_mm = False
    self.weight = nn.Parameter(
      torch.empty(out_features, in_features, dtype=compute_dtype), requires_grad=False
    )
    if bias:
      self.bias = nn.Parameter(torch.empty(out_features, dtype=compute_dtype), requires_grad=False)
    else:
      self.bias = None
    self.register_buffer("input_scale", None, persistent=False)

  def _fallback_linear(self, x: torch.Tensor) -> torch.Tensor:
    weight = self.weight
    if hasattr(weight, "dequantize"):
      weight = weight.dequantize()
    weight = weight.to(device=x.device, dtype=x.dtype)
    bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
    return F.linear(x, weight, bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.layout_type is None or self.full_precision_mm:
      return self._fallback_linear(x)

    input_shape = x.shape
    x_2d = x.reshape(-1, input_shape[-1]) if x.ndim > 2 else x
    if x_2d.ndim != 2:
      return self._fallback_linear(x)

    try:
      QuantizedTensor, _get_layout_class = _load_comfy_kitchen()
      scale = self.input_scale
      if isinstance(scale, torch.Tensor):
        scale = scale.to(device=x_2d.device)
      q_input = QuantizedTensor.from_float(x_2d, self.layout_type, scale=scale)
      out = F.linear(q_input, self.weight, self.bias)
    except Exception as exc:
      if not self._fallback_warned:
        warnings.warn(
          f"Comfy quant fast linear failed once ({exc!r}); falling back to dequantized matmul.",
          stacklevel=2,
        )
        self._fallback_warned = True
      out = self._fallback_linear(x_2d)

    if x.ndim > 2:
      out = out.reshape(*input_shape[:-1], self.out_features)
    return out


def swap_linears_to_comfy_quant(
  module: nn.Module,
  state_dict: dict[str, torch.Tensor],
  compute_dtype: torch.dtype,
  *,
  prefix: str = "",
) -> None:
  """Replace Linear layers that carry Comfy quant metadata."""
  for name, child in list(module.named_children()):
    child_prefix = f"{prefix}{name}"
    config = _decode_comfy_quant_config(state_dict.get(f"{child_prefix}{COMFY_QUANT_SUFFIX}"))
    if isinstance(child, nn.Linear) and config is not None:
      quant_format = config.get("format")
      if quant_format == COMFY_NVFP4_FORMAT:
        setattr(
          module,
          name,
          ComfyQuantLinear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            compute_dtype=compute_dtype,
            layout_type=COMFY_NVFP4_LAYOUT,
          ),
        )
      elif quant_format == COMFY_FP8_FORMAT:
        setattr(
          module,
          name,
          ComfyQuantLinear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            compute_dtype=compute_dtype,
            layout_type=COMFY_FP8_LAYOUT,
          ),
        )
      else:
        raise RuntimeError(
          f"Unsupported Comfy quantization format {quant_format!r} for {child_prefix}. "
          "Supported formats are NVFP4 and float8_e4m3fn."
        )
    else:
      swap_linears_to_comfy_quant(
        child, state_dict, compute_dtype, prefix=f"{child_prefix}."
      )


def _move_module_buffers(model: nn.Module, device: torch.device, dtype: torch.dtype) -> None:
  for name, buffer in list(model.named_buffers()):
    if buffer is None:
      continue
    parent_path, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    persistent = leaf not in parent._non_persistent_buffers_set
    if buffer.is_floating_point():
      moved = buffer.to(device=device, dtype=dtype)
    else:
      moved = buffer.to(device=device)
    if moved is not buffer:
      parent.register_buffer(leaf, moved, persistent=persistent)


def load_comfy_quant_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
  *,
  assign: bool = True,
  strict: bool = True,
) -> None:
  """Load ComfyUI mixed-precision weights for NVFP4/FP8 Linear layers."""
  QuantizedTensor, get_layout_class = _load_comfy_kitchen()
  consumed: set[str] = set()
  expected_missing: set[str] = set()

  for module_name, module in model.named_modules():
    if not isinstance(module, (ComfyQuantLinear, Fp8Linear)):
      continue

    prefix = f"{module_name}." if module_name else ""
    config_key = f"{prefix}comfy_quant"
    config = _decode_comfy_quant_config(state_dict.get(config_key))
    if config is None:
      continue
    quant_format = config.get("format")
    if quant_format == COMFY_FP8_FORMAT:
      if not isinstance(module, ComfyQuantLinear):
        raise RuntimeError(f"Layer {module_name} is not prepared for Comfy FP8 weights")
      weight_key = f"{prefix}weight"
      scale_key = f"{prefix}weight_scale"
      weight = state_dict.get(weight_key)
      scale = state_dict.get(scale_key)
      if weight is None or scale is None:
        raise RuntimeError(f"Missing Comfy FP8 tensors for quantized layer {module_name}")

      layout_cls = get_layout_class(COMFY_FP8_LAYOUT)
      if layout_cls is None:
        raise RuntimeError(f"comfy-kitchen does not provide {COMFY_FP8_LAYOUT}")

      params = layout_cls.Params(
        scale=scale.to(device=device, dtype=torch.float32),
        orig_dtype=dtype,
        orig_shape=module._orig_shape,
      )
      module.weight = nn.Parameter(
        QuantizedTensor(weight.to(device=device, dtype=FP8_WEIGHT_DTYPE), COMFY_FP8_LAYOUT, params),
        requires_grad=False,
      )
      module.full_precision_mm = bool(config.get("full_precision_matrix_mult", False))
      consumed.update({weight_key, config_key})
      consumed.add(scale_key)
      expected_missing.add(weight_key)

      bias_key = f"{prefix}bias"
      if bias_key in state_dict:
        module.bias = nn.Parameter(
          state_dict[bias_key].to(device=device, dtype=dtype), requires_grad=False
        )
        consumed.add(bias_key)
        expected_missing.add(bias_key)

      input_scale_key = f"{prefix}input_scale"
      if input_scale_key in state_dict:
        module.register_buffer(
          "input_scale",
          state_dict[input_scale_key].to(device=device, dtype=torch.float32),
          persistent=False,
        )
        consumed.add(input_scale_key)
      continue

    if quant_format != COMFY_NVFP4_FORMAT:
      raise RuntimeError(f"Unsupported Comfy quantization format {quant_format!r} at {module_name}")
    if not isinstance(module, ComfyQuantLinear):
      raise RuntimeError(f"Layer {module_name} is not prepared for NVFP4 weights")

    weight_key = f"{prefix}weight"
    block_scale_key = f"{prefix}weight_scale"
    tensor_scale_key = f"{prefix}weight_scale_2"
    weight = state_dict.get(weight_key)
    block_scale = state_dict.get(block_scale_key)
    tensor_scale = state_dict.get(tensor_scale_key)
    if weight is None or block_scale is None or tensor_scale is None:
      raise RuntimeError(f"Missing NVFP4 tensors for quantized layer {module_name}")

    layout_cls = get_layout_class(COMFY_NVFP4_LAYOUT)
    if layout_cls is None:
      raise RuntimeError(f"comfy-kitchen does not provide {COMFY_NVFP4_LAYOUT}")

    qdata = weight.to(device=device, dtype=torch.uint8)
    block_scale = block_scale.to(device=device)
    if block_scale.dtype == torch.uint8:
      block_scale = block_scale.view(dtype=torch.float8_e4m3fn)
    tensor_scale = tensor_scale.to(device=device, dtype=torch.float32)
    params = layout_cls.Params(
      scale=tensor_scale,
      block_scale=block_scale,
      orig_dtype=dtype,
      orig_shape=module._orig_shape,
    )
    module.weight = nn.Parameter(
      QuantizedTensor(qdata, COMFY_NVFP4_LAYOUT, params), requires_grad=False
    )
    module.full_precision_mm = bool(config.get("full_precision_matrix_mult", False))
    consumed.update({weight_key, block_scale_key, tensor_scale_key, config_key})
    expected_missing.add(weight_key)

    bias_key = f"{prefix}bias"
    if bias_key in state_dict:
      module.bias = nn.Parameter(
        state_dict[bias_key].to(device=device, dtype=dtype), requires_grad=False
      )
      consumed.add(bias_key)
      expected_missing.add(bias_key)

    input_scale_key = f"{prefix}input_scale"
    if input_scale_key in state_dict:
      module.register_buffer(
        "input_scale",
        state_dict[input_scale_key].to(device=device, dtype=torch.float32),
        persistent=False,
      )
      consumed.add(input_scale_key)

  prepared: dict[str, torch.Tensor] = {}
  for name, tensor in state_dict.items():
    if name in consumed:
      continue
    if name.endswith(COMFY_QUANT_SUFFIX):
      consumed.add(name)
      continue
    if (
      name.endswith(".weight")
      and tensor.dtype == FP8_WEIGHT_DTYPE
      and f"{name}_scale" in state_dict
    ):
      scale_tensor = state_dict[f"{name}_scale"].to(device=device, dtype=torch.float32)
      layout_cls = get_layout_class(COMFY_FP8_LAYOUT)
      if layout_cls is not None:
        params = layout_cls.Params(
          scale=scale_tensor,
          orig_dtype=dtype,
          orig_shape=tuple(tensor.shape),
        )
        prepared[name] = QuantizedTensor(
          tensor.to(device=device, dtype=FP8_WEIGHT_DTYPE),
          COMFY_FP8_LAYOUT,
          params,
        ).dequantize().to(dtype)
      else:
        scale = scale_tensor.to(dtype=dtype)
        if scale.ndim == 1 and tensor.ndim >= 2 and scale.shape[0] == tensor.shape[0]:
          scale = scale.view(-1, *([1] * (tensor.ndim - 1)))
        prepared[name] = tensor.to(device=device, dtype=dtype) * scale
      consumed.add(f"{name}_scale")
      continue
    if name.endswith(".weight_scale"):
      base_weight_key = name.removesuffix("_scale")
      if base_weight_key in state_dict:
        consumed.add(name)
        continue
    if tensor.is_floating_point():
      prepared[name] = tensor.to(device=device, dtype=dtype)
    else:
      prepared[name] = tensor.to(device=device)

  missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
  real_missing = [m for m in missing if m not in expected_missing]
  if unexpected:
    raise RuntimeError(f"unexpected keys after Comfy quant load: {unexpected[:10]}")
  if real_missing:
    if strict:
      raise RuntimeError(f"missing keys after Comfy quant load: {real_missing[:10]}")
    warnings.warn(f"missing keys after Comfy quant load: {real_missing[:10]}", stacklevel=2)

  _move_module_buffers(model, device, dtype)
