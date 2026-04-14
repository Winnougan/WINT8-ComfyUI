<img width="1200" height="1200" alt="Logo_Winnougan" src="https://github.com/user-attachments/assets/006583fa-3d94-4a1f-a31c-1109fbc46e8b" />

WINT8 Nodes
A self-contained suite for working with INT8 quantized diffusion models. No dependency on int88 or any other external custom node — all quantization logic is built in.
🔥 WINT8 Diffuser Loader
Loads INT8 quantized diffusion models with full control over the quantization pipeline.
Features:

Tensorwise mode — one scale per weight tensor. Fast loading, minimal memory overhead
Blockwise mode — one scale per 128×128 tile. Finer quantization granularity, better quality for models with activation outliers
Sage Attention — optional replacement of ComfyUI's attention kernel with Sage Attention for reduced VRAM usage (requires sageattn package)
QuaRot — optional Hadamard rotation to spread outliers before quantization, improving INT8 quality
On-the-fly quantization — quantize fp16/bf16 checkpoints to INT8 at load time (for non-pre-quantized models)
Model type presets — dedicated exclusion lists for flux2, z-image, chroma, wan, ltx2, qwen, ernie, hidream ensuring the right layers stay in full precision

Supported models: Flux2, Z-Image Turbo, Chroma, WAN, LTX2, Qwen, Ernie, HiDream, and any INT8 pre-quantized diffusion model
🔥 WINT8 Power LoRA Loader
Multi-LoRA loader adapted for INT8 quantized models. Uses a dynamic forward hook instead of the standard ComfyUI weight patching path, so LoRAs apply correctly without needing to dequantize and re-quantize the entire model.

Same powerful multi-row UI as the Winnougan Power LoRA Loader
Uses your existing standard LoRA files — no INT8-converted LoRAs needed
Individual on/off toggles and strength controls per LoRA
Global toggle all button
Live LoRA search dialog
Passes CLIP through unchanged


Self-Contained Architecture
The WINT8 suite ships everything it needs internally:
FilePurposewint8_quant.pyInt8TensorwiseOps + DynamicLoRAHookwint8_blockwise.pyInt8BlockwiseOps with 2D tile scalingwint8_fused_kernel.pyTriton fused INT8 kernels with torch._int_mm fallbackwint8_quarot.pyHadamard rotation (scipy optional, pure-torch fallback)
Triton kernels are used automatically when available and fall back to torch._int_mm gracefully. QuaRot falls back to pure PyTorch if scipy is not installed.

```
ComfyUI/custom_nodes/
├── winnougan_nodes/
│   ├── __init__.py
│   ├── winnougan_prompt_encoder.py
│   ├── winnougan_clip_loader.py
│   ├── winnougan_power_lora_loader.py
│   └── js/
│       ├── winnougan_prompt_encoder.js
│       ├── winnougan_clip_loader.js
│       └── winnougan_power_lora_loader.js
└── wint8_nodes/
    ├── __init__.py
    ├── wint8_diffuser_loader.py
    ├── wint8_power_lora_loader.py
    ├── wint8_quant.py
    ├── wint8_blockwise.py
    ├── wint8_fused_kernel.py
    ├── wint8_quarot.py
    └── js/
        ├── wint8_diffuser_loader.js
        └── wint8_power_lora_loader.js
```

Requirements
- ComfyUI (latest)
- PyTorch 2.1+
- `triton` (optional, for fused INT8 kernels — Linux native, Windows via `triton-windows`)
- `scipy` (optional, for optimized Hadamard matrices)
- `sageattn` (optional, for Sage Attention in WINT8 Diffuser Loader)
- ComfyUI-GGUF (optional, for GGUF CLIP loading in Winnougan CLIP Loader)
