import os
import sys
import argparse
import contextlib
import hashlib
import json
import re
import random
import time
import math
import subprocess
import threading
import shutil
from pathlib import Path
from PIL import Image

# Default models and paths
IDEOGRAM_REPO_URL = "https://github.com/ideogram-oss/ideogram4.git"
PID_REPO_URL = "https://github.com/nv-tlabs/PiD.git"
PID_WEIGHTS_REPO = "nvidia/PiD"
OUTPUT_DIR = Path(__file__).resolve().parent / "app_standalone_outputs"
DEFAULT_QUANTIZATION = "nvfp4"
QUANTIZATION_REPOS = {
    "nvfp4": "Comfy-Org/Ideogram-4",
    "fp8-nvfp4-uncond": "ideogram-ai/ideogram-4-fp8",
    "fp8": "ideogram-ai/ideogram-4-fp8",
}
QUANTIZATION_LABELS = {
    "nvfp4": "nvfp4 (fast)",
    "fp8-nvfp4-uncond": "fp8-nvfp4-uncond (balanced)",
    "fp8": "fp8 (quality)",
}
QUANTIZATION_UI_CHOICES = [
    (QUANTIZATION_LABELS[key], key) for key in QUANTIZATION_REPOS
]
QUANTIZATION_LABEL_TO_KEY = {label: key for key, label in QUANTIZATION_LABELS.items()}
NVFP4_COMFY_REPO = "Comfy-Org/Ideogram-4"
NVFP4_CONFIG_REPO = "Qwen/Qwen3-VL-8B-Instruct"
NVFP4_CONDITIONAL_FILENAME = "diffusion_models/ideogram4_nvfp4_mixed.safetensors.index.json"
NVFP4_UNCONDITIONAL_FILENAME = "diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors.index.json"
NVFP4_TEXT_ENCODER_FILENAME = "text_encoders/qwen3vl_8b_nvfp4.safetensors"
NVFP4_AUTOENCODER_FILENAME = "vae/flux2-vae.safetensors"
MAX_SEED = 2**31 - 1
PREVIEW_FORMAT_WEBP = "WebP (fast preview)"
PREVIEW_FORMAT_PNG = "PNG"
PREVIEW_FORMATS = [PREVIEW_FORMAT_WEBP, PREVIEW_FORMAT_PNG]
UPSAMPLE_IDEOGRAM_REMOTE = "Ideogram (remote)"
UPSAMPLE_GEMMA_LOCAL = "gemma-4-12B (local)"
UPSAMPLE_NONE = "None (raw prompt)"
UPSAMPLERS = [UPSAMPLE_IDEOGRAM_REMOTE, UPSAMPLE_GEMMA_LOCAL, UPSAMPLE_NONE]
IMAGE_UPSCALE_NONE = "None"
IMAGE_UPSCALE_PID = "PiD 4x latent decode"
IMAGE_UPSCALERS = [IMAGE_UPSCALE_NONE, IMAGE_UPSCALE_PID]
PID_BACKBONES = ["flux2"]
PID_CKPT_TYPES = ["2k", "2kto4k"]
PID_SCALE = 4
PID_MAX_LOW_SIDE = 1024
PID_CKPT_TYPES_BY_BACKBONE = {
    "flux2": {"2k", "2kto4k"},
}
PID_ASSET_PATTERNS = {
    ("flux2", "2k"): [
        "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/*",
        "checkpoints/flux2_ae.safetensors",
    ],
    ("flux2", "2kto4k"): [
        "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/*",
        "checkpoints/flux2_ae.safetensors",
    ],
}
PID_CHECKPOINT_SPECS = {
    ("flux2", "2k"): {
        "experiment": "PiD_res2k_sr4x_official_flux2_distill_4step",
        "checkpoint_path": "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
        "label": "Flux2 PiD 2k",
    },
    ("flux2", "2kto4k"): {
        "experiment": "PiD_res2kto4k_sr4x_official_flux2_distill_4step",
        "checkpoint_path": "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/model_ema_bf16.pth",
        "label": "Flux2 PiD 2kto4k",
    },
}
UPSAMPLE_CACHE_SCHEMA_VERSION = 2
UPSAMPLE_CACHE_PATH = Path(__file__).resolve().parent / "app_standalone_upsample_cache.json"
LOCAL_IDEOGRAM_PATCH_FILES = {
    "modeling_ideogram4.py": "app_standalone_patch_modeling_ideogram4.py",
    "pipeline_ideogram4.py": "app_standalone_patch_pipeline_ideogram4.py",
    "quantized_loading.py": "app_standalone_patch_quantized_loading.py",
    "magic_prompt.py": "app_standalone_patch_magic_prompt.py",
}
LOCAL_IDEOGRAM_API_KEY_PATH = Path(__file__).resolve().parent / "app_standalone_api_key.txt"


def default_ideogram_api_key():
    """Read the remote Ideogram upsampler key from env or a local secrets file."""
    env_key = os.environ.get("MAGIC_PROMPT_API_KEY") or os.environ.get("IDEOGRAM_API_KEY")
    if env_key:
        return env_key
    try:
        return LOCAL_IDEOGRAM_API_KEY_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def normalize_quantization(quantization):
    value = str(quantization or "").strip()
    if value in QUANTIZATION_REPOS:
        return value
    if value in QUANTIZATION_LABEL_TO_KEY:
        return QUANTIZATION_LABEL_TO_KEY[value]
    lowered = value.lower()
    for label, key in QUANTIZATION_LABEL_TO_KEY.items():
        if lowered == label.lower():
            return key
    return DEFAULT_QUANTIZATION

# Parse command line arguments first to configure environment variables and APIs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Ideogram 4 Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app on (default: 7860)")
    parser.add_argument(
        "--quantization",
        type=str,
        choices=list(QUANTIZATION_REPOS),
        default=DEFAULT_QUANTIZATION,
        help="Default Ideogram 4 weight quantization: nvfp4 (fast), fp8-nvfp4-uncond (balanced), fp8 (quality).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional custom Hugging Face model ID. Leave unset to use the selected nvfp4/fp8 repo.",
    )
    parser.add_argument(
        "--nvfp4_config_repo",
        type=str,
        default=os.environ.get("IDEOGRAM_NVFP4_CONFIG_REPO", NVFP4_CONFIG_REPO),
        help="Repo that supplies tokenizer and Qwen3-VL config for Comfy NVFP4 files.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache root. Useful for pointing servers at persistent fast storage.",
    )
    parser.add_argument(
        "--hf_offline",
        action="store_true",
        help="Use only cached Hugging Face files. Fails fast if a required model file is missing.",
    )
    parser.add_argument("--compile", action="store_true", help="Compile transformer models with torch.compile (requires PyTorch 2.0+)")
    parser.add_argument(
        "--ffn_chunk_size",
        type=int,
        default=int(os.environ.get("IDEOGRAM_FFN_CHUNK_SIZE", "0")),
        help="Chunk transformer FFN along sequence dimension. 0 disables chunking.",
    )
    parser.add_argument(
        "--rope_dtype",
        type=str,
        default=os.environ.get("IDEOGRAM_ROPE_DTYPE", "float32"),
        choices=["bfloat16", "float32"],
        help="Compute Ideogram transformer RoPE cos/sin in this dtype. Use float32 for correct output.",
    )
    parser.add_argument(
        "--cfg_one_final_steps",
        type=int,
        default=int(os.environ.get("IDEOGRAM_CFG_ONE_FINAL_STEPS", "0")),
        help="Set this many final sampling steps to CFG=1.0 and skip the unconditional branch there.",
    )
    parser.add_argument("--no_share", action="store_false", dest="share", default=True, help="Disable public Gradio share link (default: False, share is enabled by default)")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the selected Ideogram pipeline before launching the UI. By default it loads on first Generate.",
    )
    parser.add_argument(
        "--no_warmup",
        action="store_true",
        help="Deprecated compatibility flag; startup preload is off unless --preload is used.",
    )
    parser.add_argument(
        "--ideogram_api_key",
        type=str,
        default=default_ideogram_api_key(),
        help="API key for the Ideogram remote prompt upsampler (or set IDEOGRAM_API_KEY / app_standalone_api_key.txt).",
    )
    parser.add_argument(
        "--warn_on_caption_issues",
        action="store_true",
        help="Warn instead of aborting when the caption verifier flags a prompt.",
    )
    parser.add_argument(
        "--pid_repo_dir",
        type=str,
        default=os.environ.get("PID_REPO_DIR", str(Path(__file__).resolve().parent / "PiD")),
        help="Local PiD source checkout used by the optional image upscaler.",
    )
    parser.add_argument(
        "--pid_no_auto_setup",
        action="store_false",
        dest="pid_auto_setup",
        default=True,
        help="Do not auto-clone PiD or download the selected PiD checkpoint assets on first PiD upscale.",
    )
    parser.add_argument(
        "--pid_keep_ideogram_loaded",
        action="store_true",
        dest="pid_keep_ideogram_loaded",
        default=True,
        help="Deprecated compatibility flag; Ideogram and PiD are kept loaded by default.",
    )
    parser.add_argument(
        "--pid_unload_ideogram_for_pid",
        action="store_false",
        dest="pid_keep_ideogram_loaded",
        help="Unload Ideogram before PiD and unload PiD before loading Ideogram to save VRAM.",
    )
    
    # Local Gemma Inference configuration parameters
    parser.add_argument("--gemma_model_id", type=str, 
                        default=os.environ.get("GEMMA_MODEL", "google/gemma-4-12B-it"), 
                        help="Hugging Face model ID for the local Gemma upsampler (default: google/gemma-4-12B-it)")
    parser.add_argument("--gemma_assistant_model_id", type=str, 
                        default=os.environ.get("GEMMA_ASSISTANT_MODEL", "google/gemma-4-12B-it-assistant"), 
                        help="Hugging Face model ID for the MTP assistant (drafter) model to speed up generation via speculative decoding. Set to empty string to disable.")
    parser.add_argument("--gemma_torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], 
                        help="Data type to load local Gemma if not quantized (default: bfloat16)")
    parser.add_argument("--gemma_quantize", action="store_true", 
                        help="Load local Gemma in 4-bit quantized mode to save memory (if VRAM is limited)")
    parser.add_argument(
        "--gemma_max_new_tokens",
        type=int,
        default=int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "2048")),
        help="Maximum tokens for local Gemma prompt JSON generation (default: 2048).",
    )
    parser.add_argument(
        "--gemma_debug_dir",
        type=str,
        default=os.environ.get("GEMMA_DEBUG_DIR", "gemma_debug"),
        help="Directory for full raw Gemma responses when JSON parsing fails. Set empty to disable file dumps.",
    )
    parser.add_argument(
        "--gemma_enable_thinking",
        action="store_true",
        help="Enable native thinking mode for Gemma 4 (requires more tokens/time, improves reasoning).",
    )
    parser.add_argument(
        "--gemma_thinking_effort",
        type=str,
        choices=["concise", "normal", "deep"],
        default="concise",
        help="Control the length of Gemma 4 reasoning via prompt engineering.",
    )
    args = parser.parse_args()
else:
    # Fallback default args for non-CLI imports
    class Args:
        host = "0.0.0.0"
        port = 7860
        quantization = DEFAULT_QUANTIZATION
        model_id = None
        nvfp4_config_repo = os.environ.get("IDEOGRAM_NVFP4_CONFIG_REPO", NVFP4_CONFIG_REPO)
        hf_cache_dir = None
        hf_offline = False
        compile = False
        ffn_chunk_size = int(os.environ.get("IDEOGRAM_FFN_CHUNK_SIZE", "0"))
        rope_dtype = os.environ.get("IDEOGRAM_ROPE_DTYPE", "float32")
        cfg_one_final_steps = int(os.environ.get("IDEOGRAM_CFG_ONE_FINAL_STEPS", "0"))
        share = True
        preload = False
        no_warmup = False
        ideogram_api_key = default_ideogram_api_key()
        warn_on_caption_issues = False
        pid_repo_dir = os.environ.get("PID_REPO_DIR", str(Path(__file__).resolve().parent / "PiD"))
        pid_auto_setup = True
        pid_keep_ideogram_loaded = True
        gemma_model_id = os.environ.get("GEMMA_MODEL", "google/gemma-4-12B-it")
        gemma_assistant_model_id = os.environ.get("GEMMA_ASSISTANT_MODEL", "google/gemma-4-12B-it-assistant")
        gemma_torch_dtype = "bfloat16"
        gemma_quantize = False
        gemma_max_new_tokens = int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "2048"))
        gemma_debug_dir = os.environ.get("GEMMA_DEBUG_DIR", "gemma_debug")
        gemma_enable_thinking = False
        gemma_thinking_effort = "concise"
    args = Args()

# Set memory optimization environment variables
if args.hf_cache_dir:
    os.environ["HF_HOME"] = args.hf_cache_dir
if args.hf_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if not args.compile:
    # Disable torchdynamo if compiling is not requested to avoid noisy warning messages
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import gradio as gr


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ideogram4 import (  # noqa: E402
    MAGIC_PROMPTS,
    PRESETS,
    Ideogram4Pipeline,
    Ideogram4PipelineConfig,
    aspect_ratio_from_size,
)
from ideogram4.magic_prompt import build_messages, reorder_caption_keys, strip_aspect_ratio_and_bboxes, strip_aspect_ratio, strip_bboxes  # noqa: E402

# V4 presets from the official repository pipeline.
MODES = {
    "Turbo - 12 steps": "V4_TURBO_12",
    "Fast Quality - 14 steps": "V4_FAST_QUALITY_14",
    "Default - 20 steps": "V4_DEFAULT_20",
    "Quality - 48 steps": "V4_QUALITY_48",
}
ASPECT_RATIO_PRESETS = {
    "Lightning (512*512)": (512, 512),
    "Fast (1024*1024)": (1024, 1024),
    "1:1 Square": (2048, 2048),
    "16:9 Landscape": (2048, 1152),
    "9:16 Portrait": (1152, 2048),
    "4:3 Landscape": (2048, 1536),
    "3:4 Portrait": (1536, 2048),
    "3:2 Landscape": (2016, 1344),
    "2:3 Portrait": (1344, 2016),
    "21:9 Ultrawide": (2016, 864),
}
DEFAULT_ASPECT_RATIO_PRESET = "Fast (1024*1024)"

# --- Pipeline Loading ---
pipe = None
active_quantization = None
active_model_id = None
pid_decoder = None
active_pid_key = None


def enable_torch_cuda_fast_paths():
    """Enable PyTorch's built-in CUDA attention choices when available."""
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True

    for backend_flag in (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_math_sdp",
        "enable_cudnn_sdp",
    ):
        flag_setter = getattr(torch.backends.cuda, backend_flag, None)
        if flag_setter is not None:
            try:
                flag_setter(True)
            except Exception as exc:
                print(f"Warning: could not enable torch.backends.cuda.{backend_flag}: {exc}", flush=True)


def make_sdpa_context():
    """Prefer fused SDPA backends while keeping math fallback available."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backends = [
            getattr(SDPBackend, name)
            for name in (
                "CUDNN_ATTENTION",
                "FLASH_ATTENTION",
                "EFFICIENT_ATTENTION",
                "MATH",
            )
            if hasattr(SDPBackend, name)
        ]
        if not backends:
            return contextlib.nullcontext()

        try:
            return sdpa_kernel(backends, set_priority=True)
        except TypeError:
            try:
                return sdpa_kernel(backends, set_priority_order=True)
            except TypeError:
                return sdpa_kernel(backends)
    except Exception:
        return contextlib.nullcontext()


def configure_ideogram4_runtime():
    enable_torch_cuda_fast_paths()


def model_repo_for_quantization(quantization):
    if args.model_id:
        return args.model_id
    return QUANTIZATION_REPOS[quantization]


def nvfp4_config_subfolder(default_subfolder):
    repo = str(args.nvfp4_config_repo).strip().lower()
    if repo == NVFP4_CONFIG_REPO.lower():
        return ""
    return default_subfolder


def pipeline_config_for_quantization(quantization, model_id):
    if quantization == "nvfp4":
        return Ideogram4PipelineConfig(
            weights_repo=model_id,
            conditional_index_filename=NVFP4_CONDITIONAL_FILENAME,
            unconditional_index_filename=NVFP4_UNCONDITIONAL_FILENAME,
            autoencoder_filename=NVFP4_AUTOENCODER_FILENAME,
            tokenizer_repo=args.nvfp4_config_repo,
            text_encoder_config_repo=args.nvfp4_config_repo,
            tokenizer_subfolder=nvfp4_config_subfolder("tokenizer"),
            text_encoder_subfolder=nvfp4_config_subfolder("text_encoder"),
            text_encoder_weights_repo=model_id,
            text_encoder_weights_filename=NVFP4_TEXT_ENCODER_FILENAME,
        )
    if quantization == "fp8-nvfp4-uncond":
        return Ideogram4PipelineConfig(
            weights_repo=model_id,
            unconditional_weights_repo=NVFP4_COMFY_REPO,
            unconditional_index_filename=NVFP4_UNCONDITIONAL_FILENAME,
        )
    return Ideogram4PipelineConfig(weights_repo=model_id)


def rope_dtype_from_name(name):
    if str(name).lower() == "float32":
        return torch.float32
    return torch.bfloat16


def configure_pipeline_runtime_options(loaded_pipe, ffn_chunk_size=None, rope_dtype_name=None):
    chunk_size = int(ffn_chunk_size if ffn_chunk_size is not None else args.ffn_chunk_size)
    rope_dtype = rope_dtype_from_name(rope_dtype_name if rope_dtype_name is not None else args.rope_dtype)
    for transformer in (
        getattr(loaded_pipe, "conditional_transformer", None),
        getattr(loaded_pipe, "unconditional_transformer", None),
    ):
        target = getattr(transformer, "_orig_mod", transformer)
        setter = getattr(target, "set_runtime_options", None)
        if setter is not None:
            setter(ffn_chunk_size=chunk_size, rope_compute_dtype=rope_dtype)


def maybe_compile_pipeline(loaded_pipe):
    if not args.compile:
        return loaded_pipe
    print("Compiling transformer models using torch.compile... (Note: First run will take a few minutes to compile)", flush=True)
    t_comp = time.perf_counter()
    try:
        loaded_pipe.conditional_transformer = torch.compile(loaded_pipe.conditional_transformer)
        loaded_pipe.unconditional_transformer = torch.compile(loaded_pipe.unconditional_transformer)
        print(f"torch.compile setups complete in {time.perf_counter() - t_comp:.2f}s", flush=True)
    except Exception as e:
        print(f"Failed to apply torch.compile: {e}. Falling back to eager mode.", flush=True)
    return loaded_pipe


def unload_ideogram_pipeline(reason=""):
    """Release the active Ideogram pipeline and CUDA cache."""
    global pipe, active_quantization, active_model_id
    if pipe is None:
        return

    suffix = f" ({reason})" if reason else ""
    print(f"Unloading Ideogram 4 {active_quantization} pipeline{suffix}...", flush=True)
    del pipe
    pipe = None
    active_quantization = None
    active_model_id = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_pid_decoder(reason=""):
    """Release the active PiD decoder and CUDA cache."""
    global pid_decoder, active_pid_key
    if pid_decoder is None:
        return

    suffix = f" ({reason})" if reason else ""
    print(f"Unloading PiD decoder {active_pid_key}{suffix}...", flush=True)
    del pid_decoder
    pid_decoder = None
    active_pid_key = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_pipeline(quantization):
    global pipe, active_quantization, active_model_id
    quantization = normalize_quantization(quantization)
    model_id = model_repo_for_quantization(quantization)
    if pipe is not None and active_quantization == quantization and active_model_id == model_id:
        return pipe

    if pipe is not None:
        unload_ideogram_pipeline()
    if pid_decoder is not None and not args.pid_keep_ideogram_loaded:
        unload_pid_decoder("freeing VRAM for Ideogram 4")

    print(f"Loading Ideogram 4 {quantization} pipeline from model ID: {model_id}...", flush=True)
    print("Hugging Face will download missing files once and reuse its local cache on later runs.", flush=True)
    t = time.perf_counter()
    configure_ideogram4_runtime()
    loaded_pipe = Ideogram4Pipeline.from_pretrained(
        config=pipeline_config_for_quantization(quantization, model_id),
        device="cuda",
        dtype=torch.bfloat16,
    )
    configure_pipeline_runtime_options(loaded_pipe)
    pipe = maybe_compile_pipeline(loaded_pipe)
    active_quantization = quantization
    active_model_id = model_id
    print(f"Ideogram 4 {quantization} pipeline loaded on CUDA in: {time.perf_counter() - t:.1f}s", flush=True)
    return pipe


def normalize_pid_options(backbone, ckpt_type, scale, inference_steps, cfg_scale, degrade_sigma):
    backbone = (backbone or "flux").strip().lower()
    ckpt_type = (ckpt_type or "2k").strip().lower()
    if backbone not in PID_BACKBONES:
        raise ValueError(f"Unsupported PiD backbone: {backbone}. Choose one of: {', '.join(PID_BACKBONES)}")
    if ckpt_type not in PID_CKPT_TYPES_BY_BACKBONE[backbone]:
        supported = ", ".join(sorted(PID_CKPT_TYPES_BY_BACKBONE[backbone]))
        raise ValueError(f"PiD backbone '{backbone}' supports checkpoint type(s): {supported}.")

    scale = int(scale or PID_SCALE)
    if scale != PID_SCALE:
        raise ValueError(f"Ideogram PiD latent decode uses the released fixed {PID_SCALE}x PiD scale.")

    inference_steps = int(inference_steps or 4)
    if inference_steps < 1:
        raise ValueError("PiD inference steps must be at least 1.")

    cfg_scale = float(cfg_scale if cfg_scale is not None else 1.0)
    degrade_sigma = float(degrade_sigma if degrade_sigma is not None else 0.0)
    if degrade_sigma != 0.0:
        raise ValueError("Ideogram PiD latent decode uses the final clean latent, so PiD degrade sigma must be 0.0.")

    return backbone, ckpt_type, scale, inference_steps, cfg_scale, degrade_sigma


def local_pid_repo_dir():
    return Path(args.pid_repo_dir).expanduser().resolve()


def ensure_local_pid_source():
    """Return a local PiD checkout, cloning it on first optional use if allowed."""
    repo_dir = local_pid_repo_dir()
    model_loader = repo_dir / "pid" / "_src" / "utils" / "model_loader.py"
    config_file = repo_dir / "pid" / "_src" / "configs" / "pid" / "config.py"
    if model_loader.exists() and config_file.exists():
        return repo_dir

    if repo_dir.exists():
        raise RuntimeError(
            f"Found {repo_dir}, but PiD's model loader/config files are missing. "
            "Point --pid_repo_dir at a valid PiD checkout or move that directory aside."
        )

    if not args.pid_auto_setup:
        raise RuntimeError(
            f"PiD source is not installed at {repo_dir}. Clone {PID_REPO_URL} there "
            "or launch without --pid_no_auto_setup."
        )

    print(f"Local PiD source not found; cloning {PID_REPO_URL}...", flush=True)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", PID_REPO_URL, str(repo_dir)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Could not clone {PID_REPO_URL}. Install git/network access, "
            f"or clone the repo manually to {repo_dir}."
        ) from exc

    if not model_loader.exists() or not config_file.exists():
        raise RuntimeError(f"Cloned {PID_REPO_URL}, but required PiD loader/config files were not created.")
    return repo_dir


def pid_asset_pattern_present(repo_dir, pattern):
    if "*" not in pattern:
        return (repo_dir / pattern).exists()

    base = pattern.split("*", 1)[0].rstrip("/\\")
    base_path = repo_dir / base
    if not base_path.exists():
        return False
    return any(path.is_file() for path in base_path.rglob("*"))


def ensure_pid_assets(repo_dir, backbone, ckpt_type):
    patterns = PID_ASSET_PATTERNS[(backbone, ckpt_type)]
    if all(pid_asset_pattern_present(repo_dir, pattern) for pattern in patterns):
        return

    if not args.pid_auto_setup:
        raise RuntimeError(
            f"PiD assets for {backbone}/{ckpt_type} are missing under {repo_dir / 'checkpoints'}. "
            f"Run: hf download {PID_WEIGHTS_REPO} --local-dir {repo_dir} --include "
            + " ".join(f'"{pattern}"' for pattern in patterns)
        )

    print(
        f"Downloading PiD assets for {backbone}/{ckpt_type} from {PID_WEIGHTS_REPO}...",
        flush=True,
    )
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=PID_WEIGHTS_REPO,
            local_dir=str(repo_dir),
            allow_patterns=patterns,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download PiD assets for {backbone}/{ckpt_type}. "
            "Install/upgrade huggingface_hub, check Hugging Face access, or download the "
            f"patterns manually from {PID_WEIGHTS_REPO}: {', '.join(patterns)}"
        ) from exc


def caption_text_for_pid(final_prompt, source_prompt):
    """Turn Ideogram's structured caption into a compact plain-text PiD condition."""
    try:
        caption = json.loads(final_prompt)
    except Exception:
        text = final_prompt or source_prompt or ""
        return text[:4000]

    pieces = []
    high_level = caption.get("high_level_description")
    if isinstance(high_level, str):
        pieces.append(high_level)

    style = caption.get("style_description")
    if isinstance(style, dict):
        pieces.extend(value for value in style.values() if isinstance(value, str))
    elif isinstance(style, str):
        pieces.append(style)

    elements = caption.get("elements")
    if isinstance(elements, list):
        for element in elements:
            if not isinstance(element, dict):
                continue
            for key in ("description", "visual_description", "content"):
                value = element.get(key)
                if isinstance(value, str):
                    pieces.append(value)

    text = " ".join(piece.strip() for piece in pieces if piece and piece.strip())
    return (text or source_prompt or final_prompt or "")[:4000]


def get_pid_decoder(backbone, ckpt_type):
    """Load a PiD decoder checkpoint for Ideogram's Flux2 VAE latent space."""
    global pid_decoder, active_pid_key
    key = (backbone, ckpt_type)
    if pid_decoder is not None and active_pid_key == key:
        return pid_decoder
    if pid_decoder is not None:
        unload_pid_decoder()

    repo_dir = ensure_local_pid_source()
    ensure_pid_assets(repo_dir, backbone, ckpt_type)
    spec = PID_CHECKPOINT_SPECS[key]
    checkpoint_path = repo_dir / spec["checkpoint_path"]
    if not checkpoint_path.exists():
        raise RuntimeError(f"PiD checkpoint was not found after download: {checkpoint_path}")

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    try:
        from pid._src.utils.model_loader import load_model_from_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "PiD dependencies are missing. Install the optional PiD requirements from "
            "app_standalone_requirements.txt and make sure --pid_repo_dir points to PiD."
        ) from exc

    print(f"Loading {spec['label']} decoder from {checkpoint_path}...", flush=True)
    cwd = os.getcwd()
    try:
        os.chdir(repo_dir)
        model, _config = load_model_from_checkpoint(
            experiment_name=spec["experiment"],
            checkpoint_path=str(checkpoint_path),
            config_file="pid/_src/configs/pid/config.py",
            enable_fsdp=False,
            experiment_opts=[],
            strict=False,
            load_ema_to_reg=False,
        )
    finally:
        os.chdir(cwd)

    model.eval()
    pid_decoder = model
    active_pid_key = key
    print(f"{spec['label']} decoder ready.", flush=True)
    return pid_decoder


def neg1_tensor_to_pil(tensor):
    """Convert a PiD output tensor in [-1, 1] to a PIL image."""
    while tensor.dim() > 3:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif tensor.shape[0] in (3, 4):
            tensor = tensor[:, 0]
        else:
            tensor = tensor[0]
    if tensor.dim() != 3:
        raise RuntimeError(f"PiD returned an unsupported sample shape: {tuple(tensor.shape)}")
    if tensor.shape[0] not in (3, 4) and tensor.shape[-1] in (3, 4):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] not in (3, 4):
        raise RuntimeError(f"PiD returned an unsupported channel layout: {tuple(tensor.shape)}")
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    array = ((tensor.float().clamp(-1, 1) + 1.0) * 127.5)
    array = array.permute(1, 2, 0).cpu().numpy().astype("uint8")
    return Image.fromarray(array)


def patchify_flux2_raw_latents(raw_latents):
    """Flux2/PiD patchify: (B, 32, H/8, W/8) -> (B, 128, H/16, W/16)."""
    if raw_latents.ndim != 4:
        raise RuntimeError(f"Expected raw Flux2 latents with shape (B,C,H,W), got {tuple(raw_latents.shape)}")
    batch_size, channels, height, width = raw_latents.shape
    if channels != 32:
        raise RuntimeError(f"Flux2 PiD expects 32-channel raw VAE latents before patchify, got {channels}")
    if height % 2 != 0 or width % 2 != 0:
        raise RuntimeError(f"Flux2 raw latent height/width must be even before patchify, got {height}x{width}")
    return (
        raw_latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch_size, channels * 4, height // 2, width // 2)
        .contiguous()
    )


def normalize_flux2_packed_latents_with_pid_vae(pid_model, packed_latents):
    """Apply PiD's Flux2 VAE BatchNorm normalization to packed raw latents."""
    vae_interface = getattr(pid_model, "vae_encoder", None)
    vae_wrapper = getattr(vae_interface, "model", None)
    autoencoder = getattr(vae_wrapper, "model", None)
    bn = getattr(autoencoder, "bn", None)
    if bn is None:
        raise RuntimeError("Loaded PiD model does not expose a Flux2 VAE BatchNorm normalizer.")

    bn.eval()
    eps = float(getattr(autoencoder, "bn_eps", getattr(bn, "eps", 1e-4)))
    mean = bn.running_mean.view(1, -1, 1, 1).to(packed_latents.device, packed_latents.dtype)
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1).to(packed_latents.device, packed_latents.dtype) + eps)
    if mean.shape[1] != packed_latents.shape[1]:
        raise RuntimeError(
            f"PiD Flux2 VAE BN has {mean.shape[1]} channels, but packed Ideogram latents have "
            f"{packed_latents.shape[1]} channels."
        )
    return (packed_latents - mean) / std


def decode_ideogram_latents_with_pid(
    latent_output,
    final_prompt,
    source_prompt,
    backbone,
    ckpt_type,
    scale,
    inference_steps,
    cfg_scale,
    degrade_sigma,
    seed,
):
    backbone, ckpt_type, scale, inference_steps, cfg_scale, degrade_sigma = normalize_pid_options(
        backbone,
        ckpt_type,
        scale,
        inference_steps,
        cfg_scale,
        degrade_sigma,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("PiD latent decoding requires CUDA.")
    if not isinstance(latent_output, dict):
        raise RuntimeError(f"Ideogram latent output had an unexpected type: {type(latent_output)!r}")

    raw_latents = latent_output.get("latents")
    baseline = latent_output.get("decoded")
    if not isinstance(raw_latents, torch.Tensor) or raw_latents.ndim != 4:
        raise RuntimeError(f"Ideogram returned an unexpected latent shape: {getattr(raw_latents, 'shape', None)}")
    if raw_latents.shape[1] != 32:
        raise RuntimeError(f"Flux2 PiD expects 32-channel raw Ideogram VAE latents, got shape {tuple(raw_latents.shape)}")
    if not isinstance(baseline, torch.Tensor) or baseline.ndim != 4:
        raise RuntimeError(f"Ideogram returned an unexpected decoded tensor shape: {getattr(baseline, 'shape', None)}")

    lq_h, lq_w = baseline.shape[-2], baseline.shape[-1]
    if max(lq_w, lq_h) > PID_MAX_LOW_SIDE:
        raise RuntimeError(
            f"PiD 4x latent decode supports low-res sides up to {PID_MAX_LOW_SIDE}px "
            f"(requested {lq_w}x{lq_h}, output would be {lq_w * scale}x{lq_h * scale})."
        )

    pid_prompt = caption_text_for_pid(final_prompt, source_prompt)
    raw_latents = raw_latents.detach().to(dtype=torch.bfloat16, device="cuda").contiguous()

    if not args.pid_keep_ideogram_loaded:
        unload_ideogram_pipeline("freeing VRAM for PiD latent decode")

    pid_model = get_pid_decoder(backbone, ckpt_type)
    packed_latents = patchify_flux2_raw_latents(raw_latents)
    latents = normalize_flux2_packed_latents_with_pid_vae(pid_model, packed_latents).to(
        dtype=torch.bfloat16,
        device="cuda",
    )
    pid_seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, MAX_SEED)
    data_batch = {
        pid_model.config.input_caption_key: [pid_prompt],
        "LQ_latent": latents,
        "degrade_sigma": torch.tensor([degrade_sigma], device="cuda", dtype=torch.float32),
    }

    print(
        f"Running PiD latent decode: backbone={backbone}, ckpt={ckpt_type}, "
        f"scale={scale}, steps={inference_steps}, sigma={degrade_sigma}",
        flush=True,
    )
    with torch.no_grad():
        samples = pid_model.generate_samples_from_batch(
            data_batch,
            cfg_scale=float(cfg_scale),
            num_steps=int(inference_steps),
            seed=pid_seed,
            shift=None,
            image_size=(lq_h * scale, lq_w * scale),
        )
    return neg1_tensor_to_pil(samples[0])


# Lazy-loaded local Gemma models
gemma_tokenizer = None
gemma_model = None
gemma_assistant_model = None

def load_gemma_local():
    """Lazily load the local Gemma model on the first request to minimize startup memory."""
    global gemma_tokenizer, gemma_model, gemma_assistant_model
    if gemma_model is not None:
        return
        
    print(f"Loading local Gemma model '{args.gemma_model_id}'...", flush=True)
    t_gemma = time.perf_counter()
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    gemma_tokenizer = AutoTokenizer.from_pretrained(args.gemma_model_id)
    
    if not args.gemma_quantize:
        # Load in full precision (bf16/fp16/fp32)
        torch_dtype = torch.bfloat16
        if args.gemma_torch_dtype == "float16":
            torch_dtype = torch.float16
        elif args.gemma_torch_dtype == "float32":
            torch_dtype = torch.float32
            
        print(f"Loading Gemma model with dtype {torch_dtype} (no quantization)...", flush=True)
        gemma_model = AutoModelForCausalLM.from_pretrained(
            args.gemma_model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Load in optimized 4-bit mode (requires bitsandbytes) to fit into VRAM alongside Ideogram 4
        print("Loading Gemma model in 4-bit quantized mode (optimized for memory)...", flush=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        gemma_model = AutoModelForCausalLM.from_pretrained(
            args.gemma_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    print(f"Local Gemma loaded successfully in {time.perf_counter() - t_gemma:.1f}s", flush=True)

    if args.gemma_assistant_model_id:
        print(f"Loading MTP assistant model '{args.gemma_assistant_model_id}'...", flush=True)
        t_assistant = time.perf_counter()
        if not args.gemma_quantize:
            gemma_assistant_model = AutoModelForCausalLM.from_pretrained(
                args.gemma_assistant_model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            gemma_assistant_model = AutoModelForCausalLM.from_pretrained(
                args.gemma_assistant_model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        print(f"MTP assistant loaded successfully in {time.perf_counter() - t_assistant:.1f}s", flush=True)


def strip_markdown_fences(text):
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(text):
    """Drop prefixes such as 'thought' and return the first balanced JSON object."""
    text = strip_markdown_fences(text)
    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


def clean_malformed_json_caption(text):
    """Repair common almost-JSON artifacts from local chat models."""
    previous = None
    while previous != text:
        previous = text
        # Fix literal empty-string fragments Gemma often inserts after values.
        text = text.replace(',""}', "}").replace(',""]', "]")
        # Fix accidental empty strings before keys: ,""elements": -> ,"elements":
        text = re.sub(
            r'([{\[,])\s*"{2,}(?=[A-Za-z_][A-Za-z0-9_]*"\s*:)',
            r'\1"',
            text,
        )
        # Fix accidental empty-string keys before object/list close: ,"foo": "bar",""} -> ,"foo": "bar"}
        text = re.sub(r',\s*"+\s*(?=[}\]])', "", text)
        # Fix doubled double-quotes around keys: ""elements": -> "elements":
        text = re.sub(r'""([a-zA-Z0-9_]+)""?\s*:', r'"\1":', text)
    return text


def normalize_gemma_caption_object(caption, aspect_ratio):
    if not isinstance(caption, dict):
        raise TypeError(f"caption must be a JSON object, got {type(caption).__name__}")

    normalized = {}
    normalized["aspect_ratio"] = caption.get("aspect_ratio") or aspect_ratio

    high_level_description = caption.get("high_level_description")
    if isinstance(high_level_description, str):
        normalized["high_level_description"] = high_level_description

    style_description = caption.get("style_description")
    if isinstance(style_description, dict):
        allowed_style_keys = {
            "aesthetics",
            "lighting",
            "photo",
            "art_style",
            "medium",
            "color_palette",
        }
        normalized["style_description"] = {
            key: style_description[key]
            for key in style_description
            if key in allowed_style_keys
        }

    cd = caption.get("compositional_deconstruction")
    if isinstance(cd, dict):
        normalized_cd = {}
        background = cd.get("background")
        if isinstance(background, str):
            normalized_cd["background"] = background

        elements = cd.get("elements")
        if isinstance(elements, list):
            normalized_elements = []
            for element in elements:
                if not isinstance(element, dict):
                    continue

                element_type = element.get("type")
                if element_type not in {"obj", "text"}:
                    element_type = "text" if "text" in element else "obj"

                normalized_element = {"type": element_type}
                if "bbox" in element:
                    normalized_element["bbox"] = element["bbox"]
                if element_type == "text":
                    normalized_element["text"] = element.get("text", "")
                normalized_element["desc"] = element.get("desc", "")
                if "color_palette" in element:
                    normalized_element["color_palette"] = element["color_palette"]
                normalized_elements.append(normalized_element)

            normalized_cd["elements"] = normalized_elements

        normalized["compositional_deconstruction"] = normalized_cd

    return reorder_caption_keys(normalized)


def parse_gemma_caption(text, aspect_ratio):
    text = extract_json_object(text)
    candidates = [text]
    cleaned = clean_malformed_json_caption(text)
    if cleaned != text:
        candidates.append(cleaned)

    last_error = None
    for candidate in candidates:
        try:
            parsed_json = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
    else:
        raise last_error

    parsed_json = normalize_gemma_caption_object(parsed_json, aspect_ratio)
    return json.dumps(parsed_json, ensure_ascii=False, separators=(",", ":"))


def repair_gemma_caption_with_json_repair(raw_text, aspect_ratio):
    try:
        from json_repair import repair_json
    except ImportError as exc:
        raise RuntimeError(
            "Install json-repair to repair malformed local Gemma JSON: "
            "pip install json-repair"
        ) from exc

    text = clean_malformed_json_caption(extract_json_object(raw_text))
    print("[gemma] Repairing malformed JSON caption with json-repair...", flush=True)
    repaired_text = repair_json(text, ensure_ascii=False, skip_json_loads=True)
    if not repaired_text:
        raise RuntimeError("json-repair returned an empty result")

    try:
        parsed_json = json.loads(repaired_text)
    except json.JSONDecodeError as exc:
        dump_gemma_raw_response("json_repair_output", repaired_text)
        raise RuntimeError(f"json-repair returned invalid JSON: {exc}") from exc

    parsed_json = normalize_gemma_caption_object(parsed_json, aspect_ratio)
    return json.dumps(parsed_json, ensure_ascii=False, separators=(",", ":"))


def normalize_gemma_max_new_tokens(max_new_tokens=None):
    if max_new_tokens is None:
        max_new_tokens = args.gemma_max_new_tokens
    return max(512, int(max_new_tokens))


gemma_debug_dump_counter = 0


def dump_gemma_raw_response(label, text):
    global gemma_debug_dump_counter
    print(f"[gemma] BEGIN FULL RAW {label} RESPONSE ({len(text)} chars)", flush=True)
    print(text, flush=True)
    print(f"[gemma] END FULL RAW {label} RESPONSE", flush=True)

    debug_dir = (getattr(args, "gemma_debug_dir", "") or "").strip()
    if not debug_dir:
        return
    try:
        gemma_debug_dump_counter += 1
        debug_path = Path(debug_dir)
        if not debug_path.is_absolute():
            debug_path = Path(__file__).resolve().parent / debug_path
        debug_path.mkdir(parents=True, exist_ok=True)
        output_path = debug_path / f"gemma_{label}_{int(time.time())}_{gemma_debug_dump_counter}.txt"
        output_path.write_text(text, encoding="utf-8")
        print(f"[gemma] Full raw {label} response saved to: {output_path}", flush=True)
    except Exception as exc:
        print(f"[gemma] Warning: could not save full raw {label} response: {exc!r}", flush=True)


def generate_gemma_text(messages, *, max_new_tokens=None, do_sample=False, enable_thinking=False):
    max_new_tokens = normalize_gemma_max_new_tokens(max_new_tokens)
    
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking:
        chat_kwargs["enable_thinking"] = True
        do_sample = True  # Force sampling to avoid greedy deterministic loops
        
    input_text = gemma_tokenizer.apply_chat_template(messages, **chat_kwargs)
    inputs = gemma_tokenizer(input_text, return_tensors="pt").to("cuda")
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    
    if enable_thinking:
        generate_kwargs.update({"temperature": 0.4, "top_p": 0.95, "repetition_penalty": 1.1})
    elif do_sample:
        generate_kwargs.update({"temperature": 0.1, "top_p": 0.95})

    if gemma_assistant_model is not None:
        generate_kwargs["assistant_model"] = gemma_assistant_model
        
    if gemma_tokenizer.eos_token_id is not None:
        generate_kwargs["pad_token_id"] = gemma_tokenizer.eos_token_id

    with torch.no_grad():
        outputs = gemma_model.generate(**inputs, **generate_kwargs)

    input_len = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_len:]
    if generated_tokens.numel() >= max_new_tokens:
        print(
            f"[gemma] Warning: output reached max_new_tokens={max_new_tokens}; "
            "increase Gemma max new tokens if the JSON is truncated.",
            flush=True,
        )
    return gemma_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def prompt_aspect_ratio(width, height):
    width = int(width)
    height = int(height)
    d = math.gcd(width, height) or 1
    return f"{width // d}:{height // d}"


def apply_aspect_ratio_preset(preset_name):
    return ASPECT_RATIO_PRESETS.get(
        preset_name,
        ASPECT_RATIO_PRESETS[DEFAULT_ASPECT_RATIO_PRESET],
    )


def build_guidance_schedule(preset, cfg_one_final_steps=0):
    """Return preset guidance with optional final CFG=1.0 branch-skipping steps."""
    schedule = list(preset.guidance_schedule)
    final_steps = max(0, min(int(cfg_one_final_steps or 0), len(schedule)))
    for i in range(final_steps):
        schedule[i] = 1.0
    return tuple(schedule)


def build_manual_upsampler_messages(prompt, width, height):
    """Return the exact prompt-guide messages used by the local Gemma upsampler."""
    messages = build_messages("v1.txt", prompt, prompt_aspect_ratio(width, height))
    return json.dumps(messages, ensure_ascii=False, indent=2)


def local_upsample_gemma(prompt, width, height, max_new_tokens=None, enable_thinking=False, thinking_effort="concise"):
    """Rewrite the prompt into Ideogram's native JSON caption using the locally served Gemma model."""
    load_gemma_local()
    max_new_tokens = normalize_gemma_max_new_tokens(max_new_tokens)

    aspect_ratio = prompt_aspect_ratio(width, height)
    messages = build_messages("v1.txt", prompt, aspect_ratio)

    if enable_thinking:
        effort_prompts = {
            "concise": "\n\nPrioritize concise reasoning. Keep your internal thoughts very brief and output the final JSON as quickly as possible.",
            "deep": "\n\nProvide a detailed, step-by-step breakdown and explicitly consider alternative hypotheses during your thinking phase before outputting the final JSON.",
            "normal": "",
        }
        effort_instruction = effort_prompts.get(str(thinking_effort).lower(), "")
        if effort_instruction and messages:
            messages[-1]["content"] += effort_instruction

    t_gen = time.perf_counter()
    print("[gemma] Formatting chat template and tokenizing input...", flush=True)
    print(f"[gemma] Running local autoregressive inference (max_new_tokens={max_new_tokens}, enable_thinking={enable_thinking})...", flush=True)
    text = generate_gemma_text(messages, max_new_tokens=max_new_tokens, do_sample=False, enable_thinking=enable_thinking)
    print(f"[gemma] Local generation complete in {time.perf_counter() - t_gen:.2f}s", flush=True)

    # Validate JSON parsing
    try:
        return parse_gemma_caption(text, aspect_ratio)
    except Exception as e:
        print(f"[gemma] Warning: Failed to parse model output as JSON: {e!r}", flush=True)
        dump_gemma_raw_response("initial", text)
        try:
            return repair_gemma_caption_with_json_repair(text, aspect_ratio)
        except Exception as repair_error:
            raise RuntimeError(
                f"Local Gemma returned invalid JSON and json-repair failed: {repair_error}"
            ) from repair_error


def looks_like_json_caption(caption: str) -> bool:
    stripped = strip_markdown_fences(caption).lstrip()
    return stripped.startswith("{")


def normalize_caption_for_model(caption: str, width=None, height=None, strip_prompt=True) -> str:
    """Match run_inference.py: feed valid JSON with optional stripping of aspect_ratio or bboxes."""
    if not looks_like_json_caption(caption):
        return caption

    aspect_ratio = prompt_aspect_ratio(width or 1024, height or 1024)
    try:
        caption = parse_gemma_caption(caption, aspect_ratio)
    except Exception as parse_error:
        print(f"[prompt] JSON-like prompt failed strict parsing: {parse_error!r}", flush=True)
        caption = repair_gemma_caption_with_json_repair(caption, aspect_ratio)

    if strip_prompt:
        return strip_aspect_ratio_and_bboxes(caption)
    return strip_aspect_ratio(caption)


def remote_upsample_ideogram(prompt, width, height, api_key=None):
    """Rewrite the prompt using Ideogram's hosted magic-prompt API."""
    api_key = (api_key or "").strip() or args.ideogram_api_key
    if not api_key:
        raise RuntimeError(
            "Set IDEOGRAM_API_KEY, MAGIC_PROMPT_API_KEY, app_standalone_api_key.txt, "
            "or pass --ideogram_api_key."
        )
    aspect_ratio = aspect_ratio_from_size(int(width), int(height))
    magic = MAGIC_PROMPTS["ideogram-4-v1"](api_key=api_key)
    return magic.expand(prompt, aspect_ratio=aspect_ratio)


upsample_cache_lock = threading.RLock()
upsample_cache_entries = None


def upsample_cache_key(prompt, upsampler, width, height, gemma_max_new_tokens=None, gemma_enable_thinking=False, gemma_thinking_effort="concise"):
    """Build a stable cache key for the source prompt and selected upsampler."""
    payload = {
        "schema": UPSAMPLE_CACHE_SCHEMA_VERSION,
        "provider": upsampler,
        "source_prompt": str(prompt or ""),
        "enable_thinking": bool(gemma_enable_thinking),
        "thinking_effort": str(gemma_thinking_effort),
    }

    cache_material = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(cache_material.encode("utf-8")).hexdigest(), payload


def load_upsample_cache():
    """Load the persistent upsample cache once per process."""
    global upsample_cache_entries
    with upsample_cache_lock:
        if upsample_cache_entries is not None:
            return upsample_cache_entries
        upsample_cache_entries = {}
        if not UPSAMPLE_CACHE_PATH.exists():
            return upsample_cache_entries
        try:
            data = json.loads(UPSAMPLE_CACHE_PATH.read_text(encoding="utf-8"))
            if data.get("schema") != UPSAMPLE_CACHE_SCHEMA_VERSION or not isinstance(data.get("entries"), dict):
                print(f"[cache] Ignoring incompatible upsample cache: {UPSAMPLE_CACHE_PATH}", flush=True)
                return upsample_cache_entries
            upsample_cache_entries = data["entries"]
            print(f"[cache] Loaded {len(upsample_cache_entries)} upsample cache entries.", flush=True)
        except Exception as exc:
            print(f"[cache] Warning: could not load upsample cache: {exc!r}", flush=True)
        return upsample_cache_entries


def write_upsample_cache():
    """Persist the in-memory upsample cache atomically."""
    with upsample_cache_lock:
        data = {
            "schema": UPSAMPLE_CACHE_SCHEMA_VERSION,
            "entries": load_upsample_cache(),
        }
        tmp_path = UPSAMPLE_CACHE_PATH.with_suffix(UPSAMPLE_CACHE_PATH.suffix + ".tmp")
        try:
            tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(UPSAMPLE_CACHE_PATH)
        except Exception as exc:
            print(f"[cache] Warning: could not write upsample cache: {exc!r}", flush=True)


def get_cached_upsample(prompt, upsampler, width, height, gemma_max_new_tokens=None, gemma_enable_thinking=False, gemma_thinking_effort="concise"):
    cache_key, _ = upsample_cache_key(prompt, upsampler, width, height, gemma_max_new_tokens, gemma_enable_thinking, gemma_thinking_effort)
    with upsample_cache_lock:
        entry = load_upsample_cache().get(cache_key)
        if isinstance(entry, dict) and isinstance(entry.get("result"), str):
            print(f"[cache] Reusing cached upsample for {upsampler}.", flush=True)
            return entry["result"]
    return None


def store_cached_upsample(prompt, upsampler, width, height, result, gemma_max_new_tokens=None, gemma_enable_thinking=False, gemma_thinking_effort="concise"):
    cache_key, metadata = upsample_cache_key(prompt, upsampler, width, height, gemma_max_new_tokens, gemma_enable_thinking, gemma_thinking_effort)
    with upsample_cache_lock:
        load_upsample_cache()[cache_key] = {
            "created_at": int(time.time()),
            "metadata": metadata,
            "result": result,
        }
        write_upsample_cache()


def upsample_prompt(prompt, upsampler, width, height, ideogram_api_key="", gemma_max_new_tokens=None, reuse_cache=True, gemma_enable_thinking=False, gemma_thinking_effort="concise"):
    if upsampler == UPSAMPLE_NONE:
        return prompt
    if upsampler not in (UPSAMPLE_IDEOGRAM_REMOTE, UPSAMPLE_GEMMA_LOCAL):
        raise ValueError(f"Unknown prompt upsampler: {upsampler}")

    if reuse_cache:
        cached_result = get_cached_upsample(prompt, upsampler, width, height, gemma_max_new_tokens, gemma_enable_thinking, gemma_thinking_effort)
        if cached_result is not None:
            return cached_result

    if upsampler == UPSAMPLE_IDEOGRAM_REMOTE:
        final_prompt = remote_upsample_ideogram(prompt, int(width), int(height), ideogram_api_key)
    else:
        final_prompt = local_upsample_gemma(prompt, int(width), int(height), gemma_max_new_tokens, gemma_enable_thinking, gemma_thinking_effort)

    store_cached_upsample(prompt, upsampler, width, height, final_prompt, gemma_max_new_tokens, gemma_enable_thinking, gemma_thinking_effort)
    return final_prompt


def save_image_artifacts(image, preview_format, seed, width, height):
    """Save a raw PNG plus a user-selected preview image."""
    preview_format = preview_format if preview_format in PREVIEW_FORMATS else PREVIEW_FORMAT_WEBP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique = f"{time.time_ns() % 1_000_000_000:09d}"
    stem = f"ideogram4_{timestamp}_{unique}_seed{int(seed)}_{int(width)}x{int(height)}"
    raw_png_path = OUTPUT_DIR / f"{stem}.png"
    image.save(raw_png_path, format="PNG")

    if preview_format == PREVIEW_FORMAT_WEBP:
        preview_path = OUTPUT_DIR / f"{stem}_preview.webp"
        image.convert("RGB").save(preview_path, format="WEBP", quality=82, method=4)
    else:
        preview_path = raw_png_path

    return str(preview_path), str(raw_png_path)


def generate(
    prompt,
    mode="Turbo - 12 steps",
    quantization=None,
    upsampler=UPSAMPLE_GEMMA_LOCAL,
    ideogram_api_key="",
    gemma_max_new_tokens=None,
    reuse_upsample_cache=True,
    strip_prompt=True,
    gemma_enable_thinking=False,
    gemma_thinking_effort="concise",
    width=1024,
    height=1024,
    seed=0,
    randomize_seed=False,
    cfg_one_final_steps=None,
    ffn_chunk_size=None,
    rope_dtype=None,
    image_upscaler=IMAGE_UPSCALE_NONE,
    pid_backbone="flux2",
    pid_ckpt_type="2kto4k",
    pid_scale=4,
    pid_inference_steps=4,
    pid_cfg_scale=1.0,
    pid_degrade_sigma=0.0,
    preview_format=PREVIEW_FORMAT_WEBP,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed or seed < 0:
        seed = random.randint(0, MAX_SEED)

    normalized_pid_options = None
    if image_upscaler == IMAGE_UPSCALE_PID:
        normalized_pid_options = normalize_pid_options(
            pid_backbone,
            pid_ckpt_type,
            pid_scale,
            pid_inference_steps,
            pid_cfg_scale,
            pid_degrade_sigma,
        )
        if max(int(width), int(height)) > PID_MAX_LOW_SIDE:
            raise ValueError(
                f"PiD 4x latent decode supports Ideogram sizes up to {PID_MAX_LOW_SIDE}px "
                f"per side before upscale. Current size is {int(width)}x{int(height)}."
            )
    elif image_upscaler != IMAGE_UPSCALE_NONE:
        raise ValueError(f"Unknown image upscaler: {image_upscaler}")

    final_prompt = prompt
    if upsampler == UPSAMPLE_IDEOGRAM_REMOTE:
        progress(0.0, desc="Upsampling (Ideogram remote)")
        t_up = time.perf_counter()
        final_prompt = upsample_prompt(
            prompt,
            upsampler,
            int(width),
            int(height),
            ideogram_api_key=ideogram_api_key,
            gemma_max_new_tokens=gemma_max_new_tokens,
            reuse_cache=reuse_upsample_cache,
            gemma_enable_thinking=gemma_enable_thinking,
            gemma_thinking_effort=gemma_thinking_effort,
        )
        print(f"[timing] Ideogram remote upsampling finished: {time.perf_counter() - t_up:.2f}s", flush=True)
    elif upsampler == UPSAMPLE_GEMMA_LOCAL:
        progress(0.0, desc="Upsampling (Gemma local)")
        t_up = time.perf_counter()
        final_prompt = upsample_prompt(
            prompt,
            upsampler,
            int(width),
            int(height),
            ideogram_api_key=ideogram_api_key,
            gemma_max_new_tokens=gemma_max_new_tokens,
            reuse_cache=reuse_upsample_cache,
            gemma_enable_thinking=gemma_enable_thinking,
            gemma_thinking_effort=gemma_thinking_effort,
        )
        print(f"[timing] Local Gemma upsampling finished: {time.perf_counter() - t_up:.2f}s", flush=True)
    elif upsampler != UPSAMPLE_NONE:
        raise ValueError(f"Unknown prompt upsampler: {upsampler}")

    final_prompt = normalize_caption_for_model(final_prompt, int(width), int(height), strip_prompt=strip_prompt)

    # Diffusion step
    progress(0.4, desc="Generating image")
    preset = PRESETS[MODES.get(mode, MODES["Turbo - 12 steps"])]
    guidance_schedule = build_guidance_schedule(
        preset,
        cfg_one_final_steps if cfg_one_final_steps is not None else args.cfg_one_final_steps,
    )
    selected_pipe = get_pipeline(quantization or args.quantization)
    configure_pipeline_runtime_options(selected_pipe, ffn_chunk_size, rope_dtype)
    t_diff = time.perf_counter()
    step_state = {"last_time": t_diff}

    def step_callback(step: int, total_steps: int):
        now = time.perf_counter()
        elapsed = now - t_diff
        step_time = now - step_state["last_time"]
        step_state["last_time"] = now
        
        avg_step_time = elapsed / max(1, step)
        remaining = avg_step_time * (total_steps - step)
        
        desc = (
            f"Generating image (Step {step}/{total_steps}) | "
            f"Step: {step_time:.2f}s | "
            f"Elapsed: {elapsed:.1f}s | "
            f"ETA: {remaining:.1f}s"
        )
        progress(0.4 + 0.45 * (step / total_steps), desc=desc)

    with make_sdpa_context():
        if image_upscaler == IMAGE_UPSCALE_PID:
            image_or_latents = selected_pipe(
                final_prompt,
                width=int(width),
                height=int(height),
                num_steps=preset.num_steps,
                guidance_schedule=guidance_schedule,
                mu=preset.mu,
                std=preset.std,
                seed=int(seed),
                raise_on_caption_issues=upsampler != UPSAMPLE_NONE and not args.warn_on_caption_issues,
                output_type="latent",
                callback_on_step_end=step_callback,
            )
        else:
            image_or_latents = selected_pipe(
                final_prompt,
                width=int(width),
                height=int(height),
                num_steps=preset.num_steps,
                guidance_schedule=guidance_schedule,
                mu=preset.mu,
                std=preset.std,
                seed=int(seed),
                raise_on_caption_issues=upsampler != UPSAMPLE_NONE and not args.warn_on_caption_issues,
                callback_on_step_end=step_callback,
            )

    print(f"[timing] Diffusion ({mode}) complete: {time.perf_counter() - t_diff:.2f}s", flush=True)

    if image_upscaler == IMAGE_UPSCALE_PID:
        progress(0.85, desc="PiD 4x latent decode")
        t_pid = time.perf_counter()
        pid_backbone, pid_ckpt_type, pid_scale, pid_inference_steps, pid_cfg_scale, pid_degrade_sigma = normalized_pid_options
        image = decode_ideogram_latents_with_pid(
            latent_output=image_or_latents,
            final_prompt=final_prompt,
            source_prompt=prompt,
            backbone=pid_backbone,
            ckpt_type=pid_ckpt_type,
            scale=pid_scale,
            inference_steps=pid_inference_steps,
            cfg_scale=pid_cfg_scale,
            degrade_sigma=pid_degrade_sigma,
            seed=seed,
        )
        print(f"[timing] PiD latent decode complete: {time.perf_counter() - t_pid:.2f}s", flush=True)
    else:
        image = image_or_latents[0]

    # Unpack the prompt caption for display
    try:
        caption = json.loads(final_prompt)
    except Exception:
        caption = {"prompt": final_prompt}

    preview_path, raw_png_path = save_image_artifacts(image, preview_format, seed, image.width, image.height)
    return preview_path, int(seed), caption, raw_png_path


def warmup():
    """Load the default selected Ideogram pipeline before the first request."""
    get_pipeline(args.quantization)


def maybe_preload():
    """Optionally load the selected pipeline before launching the Gradio server."""
    if args.preload and not args.no_warmup:
        warmup()
        return
    print(
        "Startup model preload skipped; the selected Ideogram 4 pipeline will load on first Generate.",
        flush=True,
    )

# --- Gradio UI Layout ---
CSS = '''
.dark .gradio-container { color: var(--body-text-color); }
'''

with gr.Blocks(title="Ideogram 4 Standalone") as demo:
    gr.Markdown(
        "# Ideogram 4 Standalone Server\n"
        "Ideogram's first open-weights model — a 9.3B-parameter text-to-image foundation model running on a standalone server."
    )

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="a ginger cat wearing a tiny wizard hat reading a spellbook", lines=3)
            mode = gr.Radio(choices=list(MODES.keys()), value="Turbo - 12 steps", label="Mode (speed to quality)")
            quantization = gr.Radio(
                choices=QUANTIZATION_UI_CHOICES,
                value=normalize_quantization(args.quantization),
                label="Weight quantization",
            )
            run = gr.Button("Generate", variant="primary")
            with gr.Accordion("Advanced Settings", open=False):
                upsampler = gr.Radio(
                    choices=UPSAMPLERS,
                    value=UPSAMPLE_GEMMA_LOCAL,
                    label="Prompt upsampler",
                    info="Rewrite into Ideogram's native JSON caption using the remote Ideogram API or local Gemma.",
                )
                reuse_upsample_cache = gr.Checkbox(
                    label="Reuse cached upsample",
                    value=True,
                    info="Reuse a saved upsample for the same source prompt and provider.",
                )
                strip_prompt = gr.Checkbox(
                    label="Strip aspect ratio and bboxes",
                    value=True,
                    info="Remove aspect ratio and bounding boxes from the upsampled JSON caption for diffusion.",
                )
                ideogram_api_key = gr.Textbox(
                    label="Ideogram API key",
                    placeholder="Optional for Ideogram (remote); falls back to environment",
                    type="password",
                    value="",
                )
                gemma_max_new_tokens = gr.Number(
                    value=args.gemma_max_new_tokens,
                    precision=0,
                    label="Gemma max new tokens",
                    info="Default is 2048; increase if local Gemma truncates long structured JSON captions.",
                )
                gemma_enable_thinking = gr.Checkbox(
                    label="Gemma enable thinking",
                    value=args.gemma_enable_thinking,
                    info="Enable native thinking mode for Gemma 4 (slower, but improves layout reasoning).",
                )
                gemma_thinking_effort = gr.Dropdown(
                    choices=["concise", "normal", "deep"],
                    value=args.gemma_thinking_effort,
                    label="Gemma thinking effort",
                    info="Instructs the model to limit or extend its internal reasoning length.",
                )
                quick_aspect_ratio = gr.Radio(
                    choices=list(ASPECT_RATIO_PRESETS.keys()),
                    value=DEFAULT_ASPECT_RATIO_PRESET,
                    label="Quick aspect ratio",
                )
                with gr.Row():
                    width = gr.Slider(512, 2048, value=1024, step=16, label="Width")
                    height = gr.Slider(512, 2048, value=1024, step=16, label="Height")
                with gr.Row():
                    seed = gr.Number(label="Seed", value=0, precision=0)
                    randomize = gr.Checkbox(label="Randomize seed", value=False)
                with gr.Row():
                    cfg_one_final_steps = gr.Number(
                        value=args.cfg_one_final_steps,
                        precision=0,
                        label="CFG=1 final steps",
                        info="Speed knob. 0 = official schedule. Try Turbo 1, Default 2, Quality 3-6; higher may soften details.",
                    )
                    ffn_chunk_size = gr.Number(
                        value=args.ffn_chunk_size,
                        precision=0,
                        label="FFN chunk size",
                        info="VRAM knob, usually slower. 0 = fastest. Use 2048 or 1024 only if high-res runs hit memory pressure.",
                    )
                rope_dtype = gr.Radio(
                    choices=["float32", "bfloat16"],
                    value=args.rope_dtype if args.rope_dtype in ("float32", "bfloat16") else "float32",
                    label="RoPE dtype",
                    info="float32 is the correct baseline. bfloat16 is experimental; try it only when testing NVFP4 output.",
                )
                preview_format = gr.Radio(
                    choices=PREVIEW_FORMATS,
                    value=PREVIEW_FORMAT_WEBP,
                    label="Preview format",
                    info="WebP is faster to preview; raw PNG is always available below.",
                )
                image_upscaler = gr.Radio(
                    choices=IMAGE_UPSCALERS,
                    value=IMAGE_UPSCALE_NONE,
                    label="Image upscaler",
                    info="Optionally decode Ideogram's Flux2 VAE latents with PiD at 4x. First use may clone PiD and download assets.",
                )
                with gr.Accordion("PiD upscale", open=False):
                    pid_backbone = gr.Radio(
                        choices=PID_BACKBONES,
                        value="flux2",
                        label="PiD latent backbone",
                        info="Ideogram 4 uses the Flux2 VAE latent space.",
                    )
                    pid_ckpt_type = gr.Radio(
                        choices=PID_CKPT_TYPES,
                        value="2kto4k",
                        label="PiD checkpoint",
                        info="2kto4k uses NVIDIA's fixed Flux2 _2606 checkpoint.",
                    )
                    with gr.Row():
                        pid_scale = gr.Number(value=PID_SCALE, precision=0, label="PiD scale")
                        pid_inference_steps = gr.Number(value=4, precision=0, label="PiD steps")
                    with gr.Row():
                        pid_cfg_scale = gr.Number(value=1.0, label="PiD CFG")
                        pid_degrade_sigma = gr.Slider(
                            0.0,
                            1.0,
                            value=0.0,
                            step=0.05,
                            label="PiD sigma",
                            info="Fixed at 0 for Ideogram's final clean latent.",
                            interactive=False,
                        )
                with gr.Accordion("Manual JSON", open=False):
                    build_manual_prompt = gr.Button("Build Prompt Messages")
                    manual_prompt_messages = gr.Textbox(
                        label="build_messages output",
                        lines=18,
                        max_lines=40,
                    )
        with gr.Column():
            out_image = gr.Image(label="Output Image", type="filepath", format="webp")
            raw_png_download = gr.File(label="Raw PNG download", file_types=[".png"])
            out_caption = gr.JSON(label="Caption fed to the model (upsampled when enabled)")

    gr.Examples(
        examples=[
            ["a ginger cat wearing a tiny wizard hat reading a spellbook"],
            ["an isometric illustration of a tiny city floating in the clouds"],
            ["a golden retriever on a skateboard"],
        ],
        inputs=[prompt],
        outputs=[out_image, seed, out_caption, raw_png_download],
        fn=generate,
        cache_examples=False,
    )

    quick_aspect_ratio.change(
        apply_aspect_ratio_preset,
        inputs=[quick_aspect_ratio],
        outputs=[width, height],
    )

    run.click(
        generate,
        inputs=[
            prompt,
            mode,
            quantization,
            upsampler,
            ideogram_api_key,
            gemma_max_new_tokens,
            reuse_upsample_cache,
            strip_prompt,
            gemma_enable_thinking,
            gemma_thinking_effort,
            width,
            height,
            seed,
            randomize,
            cfg_one_final_steps,
            ffn_chunk_size,
            rope_dtype,
            image_upscaler,
            pid_backbone,
            pid_ckpt_type,
            pid_scale,
            pid_inference_steps,
            pid_cfg_scale,
            pid_degrade_sigma,
            preview_format,
        ],
        outputs=[out_image, seed, out_caption, raw_png_download],
    )

    build_manual_prompt.click(
        build_manual_upsampler_messages,
        inputs=[prompt, width, height],
        outputs=[manual_prompt_messages],
    )

if __name__ == "__main__":
    maybe_preload()

    # Launch Gradio server
    print(f"Launching Gradio app on {args.host}:{args.port}...", flush=True)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=CSS,
    )
