"""
WINT8 Diffuser Loader
─────────────────────
Loads INT8 tensorwise quantized diffusion models with optional:
  • Sage Attention          — memory-efficient attention
  • QuaRot Hadamard rotation — better quantization quality
  • On-the-fly quantization  — quantize fp16/bf16 weights at load time

Wraps the int88 Int8TensorwiseOps custom operations class.
"""

import logging
import folder_paths
import comfy.sd

log = logging.getLogger("WINT8")

NODE_NAME = "WINT8 Diffuser Loader"

# Model-type exclusion lists — layers that should NOT be quantized
_EXCLUSIONS = {
    "flux2": [
        "img_in", "time_in", "guidance_in", "txt_in", "final_layer",
        "double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation",
    ],
    "z-image": [
        "cap_embedder", "t_embedder", "x_embedder", "cap_pad_token",
        "context_refiner", "final_layer", "noise_refiner", "adaLN",
        "x_pad_token", "layers.0.",
    ],
    "chroma": [
        "distilled_guidance_layer", "final_layer", "img_in", "txt_in",
        "nerf_image_embedder", "nerf_blocks", "nerf_final_layer_conv",
        "__x0__", "nerf_final_layer_conv",
    ],
    "wan": [
        "patch_embedding", "text_embedding", "time_embedding",
        "time_projection", "head", "img_emb",
    ],
    "ltx2": [
        "adaln_single", "audio_adaln_single", "audio_caption_projection",
        "audio_patchify_proj", "audio_proj_out", "audio_scale_shift_table",
        "av_ca_a2v_gate_adaln_single", "av_ca_audio_scale_shift_adaln_single",
        "av_ca_v2a_gate_adaln_single", "av_ca_video_scale_shift_adaln_single",
        "caption_projection", "patchify_proj", "proj_out", "scale_shift_table",
    ],
    "qwen": [
        "time_text_embed", "img_in", "norm_out", "proj_out", "txt_in",
    ],
    "ernie": [
        "time", "x_embedder", "adaLN", "final", "text_proj",
        "norm", "layers.0.", "layers.35",
    ],
    "hidream": [
        "patch_embedding", "time_text_embed", "norm_out", "proj_out",
    ],
    "auto": [],   # No exclusions — let Int8TensorwiseOps decide from checkpoint
}

MODEL_TYPES = list(_EXCLUSIONS.keys())


def _try_enable_sage_attention():
    """Attempt to patch ComfyUI's attention to use Sage Attention."""
    try:
        import comfy.ldm.modules.attention as attn_mod
        import sageattn  # noqa: F401  (just confirm it's importable)
        from sageattn import sageattn as _sageattn

        _orig_attn = getattr(attn_mod, "_orig_attn_wint8", None)
        if _orig_attn is not None:
            return True  # already patched

        # Patch optimized_attention
        _orig = attn_mod.optimized_attention

        def _sage_attention(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
            try:
                import torch
                # Sage Attention expects (B, H, S, D) — ComfyUI uses (B, S, H*D)
                B, S, _ = q.shape
                D = q.shape[-1] // heads
                q_ = q.view(B, S, heads, D).transpose(1, 2).contiguous()
                k_ = k.view(B, S, heads, D).transpose(1, 2).contiguous()
                v_ = v.view(B, S, heads, D).transpose(1, 2).contiguous()
                out = _sageattn(q_, k_, v_, tensor_layout="HND", is_causal=False)
                out = out.transpose(1, 2).reshape(B, S, heads * D)
                return out
            except Exception:
                return _orig(q, k, v, heads, mask, attn_precision, skip_reshape)

        attn_mod._orig_attn_wint8 = _orig
        attn_mod.optimized_attention = _sage_attention
        log.info(f"[{NODE_NAME}] Sage Attention enabled.")
        return True
    except ImportError:
        log.warning(f"[{NODE_NAME}] sageattn not installed — Sage Attention skipped.")
        return False
    except Exception as e:
        log.warning(f"[{NODE_NAME}] Sage Attention patch failed: {e}")
        return False


def _disable_sage_attention():
    """Restore original attention if previously patched."""
    try:
        import comfy.ldm.modules.attention as attn_mod
        orig = getattr(attn_mod, "_orig_attn_wint8", None)
        if orig is not None:
            attn_mod.optimized_attention = orig
            del attn_mod._orig_attn_wint8
            log.info(f"[{NODE_NAME}] Sage Attention disabled.")
    except Exception:
        pass


class WINT8DiffuserLoader:
    NAME     = NODE_NAME
    CATEGORY = "WINT8"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "model_type": (MODEL_TYPES, {"default": "flux2"}),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp16", "bf16"],
                    {"default": "default"},
                ),
                "on_the_fly_quantization": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Quantize fp16/bf16 weights to INT8 at load time. "
                        "Only needed if your checkpoint is NOT already INT8."
                    ),
                }),
                "enable_quarot": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Apply Hadamard (QuaRot) rotation to reduce outliers. "
                        "Improves quality for heavily quantized models."
                    ),
                }),
                "sage_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Replace ComfyUI's attention kernel with Sage Attention "
                        "for reduced VRAM usage. Requires sageattn package."
                    ),
                }),
                "quant_mode": (
                    ["tensorwise", "blockwise"],
                    {
                        "default": "tensorwise",
                        "tooltip": (
                            "tensorwise: one scale per weight tensor (faster load, lower memory). "
                            "blockwise: one scale per 128×128 tile (finer granularity, better quality)."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "load_diffuser"

    def load_diffuser(
        self,
        unet_name,
        model_type,
        weight_dtype,
        on_the_fly_quantization,
        enable_quarot,
        sage_attention,
        quant_mode="tensorwise",
    ):
        # ── Sage Attention ────────────────────────────────────────────────────
        if sage_attention:
            _try_enable_sage_attention()
        else:
            _disable_sage_attention()

        # ── INT8 ops setup ────────────────────────────────────────────────────
        exclusions = _EXCLUSIONS.get(model_type, [])

        if quant_mode == "blockwise":
            try:
                from .wint8_blockwise import Int8BlockwiseOps as OpsClass
            except ImportError as e:
                raise RuntimeError(f"[{NODE_NAME}] wint8_blockwise not found: {e}")
            OpsClass.excluded_names      = exclusions
            OpsClass.dynamic_quantize    = on_the_fly_quantization
            OpsClass.enable_quarot       = enable_quarot
            OpsClass.use_triton          = True
            OpsClass._is_prequantized    = False
        else:
            try:
                from .wint8_quant import Int8TensorwiseOps as OpsClass
            except ImportError as e:
                raise RuntimeError(f"[{NODE_NAME}] wint8_quant not found: {e}")
            OpsClass.excluded_names      = exclusions
            OpsClass.dynamic_quantize    = on_the_fly_quantization
            OpsClass.enable_quarot       = enable_quarot
            OpsClass.use_triton          = True
            OpsClass._is_prequantized    = False

        model_options = {"custom_operations": OpsClass}

        # Optional weight dtype override (fp8, fp16, bf16 for non-int8 layers)
        if weight_dtype != "default":
            import torch
            _dtype_map = {
                "fp16":       torch.float16,
                "bf16":       torch.bfloat16,
                "fp8_e4m3fn": torch.float8_e4m3fn,
            }
            model_options["dtype"] = _dtype_map[weight_dtype]

        # ── Load ──────────────────────────────────────────────────────────────
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        log.info(
            f"[{NODE_NAME}] Loaded '{unet_name}' | type={model_type} "
            f"| mode={quant_mode} | quarot={enable_quarot} "
            f"| otf_quant={on_the_fly_quantization} | sage={sage_attention}"
        )
        return (model,)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "WINT8DiffuserLoader": WINT8DiffuserLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WINT8DiffuserLoader": "WINT8 Diffuser Loader",
}
