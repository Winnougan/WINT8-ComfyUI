"""
WINT8 Power LoRA Loader
───────────────────────
Multi-LoRA loader that routes each LoRA through the INT8-aware
DynamicLoRAHook so they apply correctly to INT8 quantized diffusion models.

Mirrors the Winnougan Power LoRA Loader UI (same PowerLoraWidget / FlexibleOptional
pattern) but uses the dynamic transformer_options hook path instead of the
standard ComfyUI patch path, which doesn't understand INT8 weights.

Each lora row widget sends:
  { on, lora, strength, strengthTwo }   — standard dict payload
The Python side reads these and chains them through INT8DynamicLoraLoader logic.
"""

import logging
import folder_paths
import comfy.utils
import comfy.lora

log = logging.getLogger("WINT8")

NODE_NAME = "WINT8 Power LoRA Loader"


class FlexibleOptionalInputType(dict):
    """Allows any key as a valid optional input (mirrors winnougan_power_lora_loader)."""
    def __init__(self, type_, data=None):
        super().__init__(data or {})
        self.type = type_

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        if key in self.keys():
            return super().__getitem__(key)
        return (self.type,)


def _get_lora_by_filename(filename, log_node=None):
    if not filename or filename == "None":
        return None
    loras = folder_paths.get_filename_list("loras")
    for lora in loras:
        if lora == filename or lora.endswith(filename):
            return lora
    if log_node:
        log.warning(f"[{log_node}] Could not find lora: {filename}")
    return None


def _apply_single_lora(model_patcher, lora_name, strength):
    """
    Load one LoRA and register it via the DynamicLoRAHook path.
    Returns the updated model_patcher.
    """
    from .wint8_quant import DynamicLoRAHook

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_path is None:
        log.warning(f"[{NODE_NAME}] Cannot find lora file: {lora_name}")
        return model_patcher

    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

    # Build key map
    key_map = {}
    try:
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)
    except Exception:
        pass

    patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=False)
    del lora

    # Register the global hook on this diffusion model (idempotent)
    DynamicLoRAHook.register(model_patcher.model.diffusion_model)

    # Add to transformer_options dynamic_loras list (non-sticky, clone-safe)
    if "transformer_options" not in model_patcher.model_options:
        model_patcher.model_options["transformer_options"] = {}

    opts = model_patcher.model_options["transformer_options"]
    if "dynamic_loras" not in opts:
        opts["dynamic_loras"] = []
    else:
        opts["dynamic_loras"] = list(opts["dynamic_loras"])  # shallow copy

    opts["dynamic_loras"].append({
        "name":     lora_name,
        "strength": strength,
        "patches":  patch_dict,
    })

    log.info(f"[{NODE_NAME}] Registered '{lora_name}' strength={strength:.3f}")
    return model_patcher


class WINT8PowerLoraLoader:
    NAME     = NODE_NAME
    CATEGORY = "WINT8"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(type_="*", data={
                "model": ("MODEL",),
                "clip":  ("CLIP",),
            }),
            "hidden": {},
        }

    RETURN_TYPES  = ("MODEL", "CLIP")
    RETURN_NAMES  = ("MODEL", "CLIP")
    FUNCTION      = "load_loras"

    def load_loras(self, model=None, clip=None, **kwargs):
        if model is None:
            raise ValueError(f"[{NODE_NAME}] No model connected.")

        model_patcher = model.clone()

        # Iterate lora_N widget payloads (same dict format as Winnougan Power LoRA)
        for key, value in sorted(kwargs.items()):
            if not key.upper().startswith("LORA_"):
                continue

            # ── Dict payload (normal widget) ──────────────────────────────────
            if isinstance(value, dict):
                if not all(k in value for k in ("on", "lora", "strength")):
                    continue
                if not value["on"]:
                    continue

                lora_filename = value.get("lora")
                strength      = float(value.get("strength", 1.0))

                if not lora_filename or lora_filename == "None":
                    continue
                if strength == 0.0:
                    continue

                resolved = _get_lora_by_filename(lora_filename, log_node=NODE_NAME)
                if resolved is None:
                    continue

                model_patcher = _apply_single_lora(model_patcher, resolved, strength)

            # ── String payload (wired filename) ───────────────────────────────
            elif isinstance(value, str) and value and value != "None":
                resolved = _get_lora_by_filename(value, log_node=NODE_NAME)
                if resolved is None:
                    continue
                model_patcher = _apply_single_lora(model_patcher, resolved, 1.0)

        return (model_patcher, clip)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "WINT8PowerLoraLoader": WINT8PowerLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WINT8PowerLoraLoader": "WINT8 Power LoRA Loader",
}
