"""
WINT8 Node Suite for ComfyUI
─────────────────────────────
Nodes optimized for working with INT8 quantized diffusion models.

  WINT8 Diffuser Loader   — Load INT8 models with Sage Attention & QuaRot
  WINT8 Power LoRA Loader — Multi-LoRA loader using the INT8 dynamic hook path
"""

from .wint8_diffuser_loader import (
    NODE_CLASS_MAPPINGS as DIFFUSER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as DIFFUSER_DISPLAY,
)

from .wint8_power_lora_loader import (
    NODE_CLASS_MAPPINGS as POWER_LORA_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as POWER_LORA_DISPLAY,
)

NODE_CLASS_MAPPINGS = {
    **DIFFUSER_MAPPINGS,
    **POWER_LORA_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **DIFFUSER_DISPLAY,
    **POWER_LORA_DISPLAY,
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
