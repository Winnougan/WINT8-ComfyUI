"""
WINT8 CLIP Loader
─────────────────
Robust INT8 text encoder loader for ComfyUI.

ComfyUI's standard CLIP loaders don't handle INT8 quantized text encoders
well — they either ignore the quantization or fail silently and load garbage.
This node fixes that by:

  1. Loading the state dict manually before passing it to ComfyUI
  2. Detecting INT8 quantization from tensor dtypes in the state dict
  3. Injecting Int8TensorwiseOps as custom_operations when INT8 is detected
  4. Falling back gracefully to standard loading if INT8 ops aren't available
  5. Supporting single and dual CLIP loading with per-encoder dtype control

Supports all CLIP types ComfyUI knows about, including flux2, wan, ltxv,
hidream, chroma, sd3, etc.
"""

import logging
import torch
import folder_paths
import comfy.sd
import comfy.utils

log = logging.getLogger("WINT8")

NODE_NAME = "WINT8 CLIP Loader"

CLIP_TYPES = [
    "stable_diffusion", "stable_cascade", "sd3", "stable_audio",
    "mochi", "ltxv", "pixart", "cosmos", "anima", "lumina2", "wan",
    "hidream", "chroma", "ace", "qwen_image", "flux2", "ovis", "longcat_image",
    "flux", "hunyuan_video", "hunyuan_image",
]

LOAD_MODES = ["auto", "int8", "standard"]


def _get_clip_files():
    """Gather all CLIP/text encoder files from known folder keys."""
    seen, all_files = set(), []
    for key in ("clip", "text_encoders", "clip_vision", "unet_gguf"):
        try:
            for f in folder_paths.get_filename_list(key):
                if f not in seen:
                    seen.add(f)
                    all_files.append(f)
        except Exception:
            pass
    return sorted(all_files)


def _resolve_clip_path(name):
    """Find the full path for a CLIP file across all known folder keys."""
    for key in ("clip", "text_encoders", "clip_vision", "unet_gguf"):
        try:
            p = folder_paths.get_full_path(key, name)
            if p:
                return p
        except Exception:
            pass
    return None


def _resolve_clip_type(clip_type_str):
    """Convert clip type string to CLIPType enum, with clear error on unknown."""
    key = clip_type_str.upper()
    val = getattr(comfy.sd.CLIPType, key, None)
    if val is None:
        # Try common aliases
        aliases = {
            "FLUX": "FLUX",
            "FLUX2": "FLUX2",
            "SD": "STABLE_DIFFUSION",
            "SDXL": "STABLE_DIFFUSION",
        }
        key = aliases.get(key, key)
        val = getattr(comfy.sd.CLIPType, key, None)
    if val is None:
        available = [e.name.lower() for e in comfy.sd.CLIPType]
        log.warning(
            f"[{NODE_NAME}] Unknown clip_type '{clip_type_str}'. "
            f"Available: {available}. Falling back to stable_diffusion."
        )
        val = comfy.sd.CLIPType.STABLE_DIFFUSION
    return val


def _detect_int8(state_dict: dict) -> bool:
    """Return True if the state dict contains INT8 weight tensors."""
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.int8:
            return True
    return False


def _get_int8_ops():
    """Try to get Int8TensorwiseOps from wint8_quant. Returns None if unavailable."""
    try:
        from .wint8_quant import Int8TensorwiseOps
        return Int8TensorwiseOps
    except ImportError:
        try:
            # Try finding it in sys.modules if wint8_quant is loaded elsewhere
            import sys
            for mod in sys.modules.values():
                ops = getattr(mod, "Int8TensorwiseOps", None)
                if ops is not None:
                    return ops
        except Exception:
            pass
    return None


def _build_model_options(state_dict: dict, load_mode: str) -> dict:
    """
    Build model_options for ComfyUI's text encoder loader.

    - auto:     detect INT8 from state dict, use Int8TensorwiseOps if found
    - int8:     always use Int8TensorwiseOps (even for fp16 checkpoints, useful
                for on-the-fly quantization if dynamic_quantize is enabled)
    - standard: skip INT8 ops entirely, use ComfyUI's default
    """
    model_options = {}

    use_int8 = False
    if load_mode == "int8":
        use_int8 = True
    elif load_mode == "auto":
        use_int8 = _detect_int8(state_dict)
        if use_int8:
            log.info(f"[{NODE_NAME}] INT8 tensors detected — using Int8TensorwiseOps.")

    if use_int8:
        ops = _get_int8_ops()
        if ops is not None:
            # Configure the ops class for text encoder use
            # Text encoders don't need exclusion lists like the diffuser does
            ops.excluded_names   = []
            ops.dynamic_quantize = False
            ops.enable_quarot    = False
            ops.use_triton       = True
            ops._is_prequantized = False
            model_options["custom_operations"] = ops
            log.info(f"[{NODE_NAME}] Int8TensorwiseOps injected for CLIP loading.")
        else:
            log.warning(
                f"[{NODE_NAME}] load_mode='{load_mode}' requested but "
                "Int8TensorwiseOps not available (wint8_quant not found). "
                "Falling back to standard loading."
            )

    return model_options


def _load_single_clip(
    clip_name: str,
    clip_type_str: str,
    load_mode: str,
) -> object:
    """
    Load a single text encoder robustly.

    Strategy:
      1. Load the state dict manually
      2. Detect/configure INT8 ops
      3. Use load_text_encoder_state_dicts (accepts pre-loaded state dicts)
      4. On failure, attempt standard load_clip as fallback
    """
    path = _resolve_clip_path(clip_name)
    if path is None:
        raise FileNotFoundError(f"[{NODE_NAME}] Cannot find CLIP: '{clip_name}'")

    clip_type = _resolve_clip_type(clip_type_str)

    # ── Load state dict ───────────────────────────────────────────────────────
    log.info(f"[{NODE_NAME}] Loading state dict: {clip_name}")
    try:
        sd = comfy.utils.load_torch_file(path, safe_load=True)
    except Exception as e:
        raise RuntimeError(f"[{NODE_NAME}] Failed to load '{clip_name}': {e}") from e

    # ── Build model options ───────────────────────────────────────────────────
    model_options = _build_model_options(sd, load_mode)

    # ── Attempt load via state dict API ──────────────────────────────────────
    try:
        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[sd],
            clip_type=clip_type,
            model_options=model_options,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        log.info(f"[{NODE_NAME}] Loaded '{clip_name}' via state_dict path.")
        return clip
    except Exception as e:
        log.warning(
            f"[{NODE_NAME}] State dict path failed for '{clip_name}': {e}\n"
            "Attempting standard load_clip fallback..."
        )

    # ── Fallback: standard load_clip ──────────────────────────────────────────
    try:
        clip = comfy.sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )
        log.info(f"[{NODE_NAME}] Loaded '{clip_name}' via fallback load_clip.")
        return clip
    except Exception as e2:
        raise RuntimeError(
            f"[{NODE_NAME}] Both loading paths failed for '{clip_name}'.\n"
            f"  State dict error: {e}\n"
            f"  Fallback error:   {e2}"
        ) from e2


def _load_dual_clip(
    clip_name_1: str, clip_type_1: str, load_mode_1: str,
    clip_name_2: str, clip_type_2: str, load_mode_2: str,
) -> object:
    """
    Load two text encoders and merge them.

    ComfyUI merges dual CLIPs by passing both state dicts to
    load_text_encoder_state_dicts simultaneously. This works for models like
    flux (clip-l + t5), sd3, sdxl etc.

    If the two encoders have different load modes (e.g. one INT8, one standard)
    we detect per-state-dict and use INT8 ops when either requires them.
    """
    path1 = _resolve_clip_path(clip_name_1)
    path2 = _resolve_clip_path(clip_name_2)
    if path1 is None:
        raise FileNotFoundError(f"[{NODE_NAME}] Cannot find CLIP 1: '{clip_name_1}'")
    if path2 is None:
        raise FileNotFoundError(f"[{NODE_NAME}] Cannot find CLIP 2: '{clip_name_2}'")

    clip_type = _resolve_clip_type(clip_type_1)  # primary type governs merge

    log.info(f"[{NODE_NAME}] Loading dual state dicts: {clip_name_1} + {clip_name_2}")
    sd1 = comfy.utils.load_torch_file(path1, safe_load=True)
    sd2 = comfy.utils.load_torch_file(path2, safe_load=True)

    # Use INT8 ops if either encoder needs them
    needs_int8 = (
        load_mode_1 == "int8"
        or load_mode_2 == "int8"
        or (load_mode_1 == "auto" and _detect_int8(sd1))
        or (load_mode_2 == "auto" and _detect_int8(sd2))
    )

    model_options = {}
    if needs_int8:
        ops = _get_int8_ops()
        if ops is not None:
            ops.excluded_names   = []
            ops.dynamic_quantize = False
            ops.enable_quarot    = False
            ops.use_triton       = True
            ops._is_prequantized = False
            model_options["custom_operations"] = ops
            log.info(f"[{NODE_NAME}] INT8 ops injected for dual CLIP loading.")
        else:
            log.warning(f"[{NODE_NAME}] INT8 ops unavailable for dual load — using standard.")

    # ── Attempt dual state dict load ──────────────────────────────────────────
    try:
        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[sd1, sd2],
            clip_type=clip_type,
            model_options=model_options,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        log.info(f"[{NODE_NAME}] Dual CLIP loaded successfully.")
        return clip
    except Exception as e:
        log.warning(
            f"[{NODE_NAME}] Dual state dict load failed: {e}\n"
            "Attempting fallback: load each CLIP separately then merge via paths..."
        )

    # ── Fallback: pass both paths to load_clip ────────────────────────────────
    try:
        clip = comfy.sd.load_clip(
            ckpt_paths=[path1, path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )
        log.info(f"[{NODE_NAME}] Dual CLIP loaded via path fallback.")
        return clip
    except Exception as e2:
        raise RuntimeError(
            f"[{NODE_NAME}] Both dual-load strategies failed.\n"
            f"  State dict error: {e}\n"
            f"  Path fallback error: {e2}"
        ) from e2


# ── Node ──────────────────────────────────────────────────────────────────────

class WINT8CLIPLoader:
    NAME     = NODE_NAME
    CATEGORY = "WINT8"

    @classmethod
    def INPUT_TYPES(cls):
        files = _get_clip_files()
        if not files:
            files = ["none"]
        return {
            "required": {
                "clip_name_1": (files,),
                "clip_type_1": (CLIP_TYPES, {"default": "flux2"}),
                "load_mode_1": (LOAD_MODES, {
                    "default": "auto",
                    "tooltip": (
                        "auto: detect INT8 from file and use INT8 ops if found. "
                        "int8: always use INT8 ops. "
                        "standard: skip INT8 ops (use for fp16/bf16 encoders)."
                    ),
                }),
                "dual_clip": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "dual",
                    "label_off": "single",
                    "tooltip": "Load a second text encoder and merge with the first.",
                }),
                "clip_name_2": (files,),
                "clip_type_2": (CLIP_TYPES, {"default": "flux2"}),
                "load_mode_2": (LOAD_MODES, {"default": "auto"}),
            }
        }

    RETURN_TYPES  = ("CLIP",)
    RETURN_NAMES  = ("clip",)
    FUNCTION      = "load_clip"

    def load_clip(
        self,
        clip_name_1, clip_type_1, load_mode_1,
        dual_clip,
        clip_name_2, clip_type_2, load_mode_2,
    ):
        if not dual_clip:
            clip = _load_single_clip(clip_name_1, clip_type_1, load_mode_1)
            return (clip,)

        clip = _load_dual_clip(
            clip_name_1, clip_type_1, load_mode_1,
            clip_name_2, clip_type_2, load_mode_2,
        )
        return (clip,)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "WINT8CLIPLoader": WINT8CLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WINT8CLIPLoader": "WINT8 CLIP Loader",
}
