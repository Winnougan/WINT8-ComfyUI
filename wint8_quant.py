"""
wint8_quant.py
──────────────
Self-contained INT8 quantization ops for WINT8 nodes.

Provides:
  - Quantization utilities
  - Int8TensorwiseOps  — ComfyUI custom operations class for INT8 model loading
  - DynamicLoRAHook    — Non-sticky, non-patching dynamic LoRA via forward hook

No dependency on int88 or any other external custom node.
"""

import logging
import torch
import torch.nn.functional as F
from torch import Tensor, nn

log = logging.getLogger("WINT8")

# ── Triton / fallback setup ───────────────────────────────────────────────────

try:
    from .wint8_fused_kernel import (
        triton_int8_linear,
        triton_int8_linear_per_row,
        triton_quantize_rowwise,
    )
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    log.warning("WINT8: Triton not available, falling back to torch._int_mm")

# Runtime toggle — can be flipped by the loader node
_use_triton = True


# =============================================================================
# Quantization utilities
# =============================================================================

def quantize_int8(x: Tensor, scale) -> Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_int8_tensorwise(x: Tensor):
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def quantize_int8_axiswise(x: Tensor, dim: int):
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def dequantize(q: Tensor, scale) -> Tensor:
    return q.float() * scale


# =============================================================================
# Forward helpers
# =============================================================================

@torch.no_grad()
def int8_forward_dynamic(x, weight, weight_scale, bias, compute_dtype):
    """W8A8 forward with dynamic per-token activation quantization (scalar weight scale)."""
    if _TRITON_AVAILABLE and _use_triton and x.is_cuda:
        return triton_int8_linear(x, weight, weight_scale, bias, compute_dtype)
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    out = res.float().mul_(weight_scale * x_scale).to(compute_dtype)
    if bias is not None:
        out = out + bias.to(compute_dtype)
    return out


@torch.no_grad()
def int8_forward_dynamic_per_row(x, weight, weight_scale, bias, compute_dtype):
    """W8A8 forward with per-row weight scales."""
    if _TRITON_AVAILABLE and _use_triton and x.is_cuda:
        return triton_int8_linear_per_row(x, weight, weight_scale, bias, compute_dtype)
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    out = res.float().mul_(x_scale).mul_(weight_scale.T).to(compute_dtype)
    if bias is not None:
        out = out + bias.to(compute_dtype)
    return out


# =============================================================================
# Dynamic LoRA Hook
# =============================================================================

class DynamicLoRAHook:
    """
    Registered on diffusion_model.  At the start of each forward pass it reads
    transformer_options["dynamic_loras"] and composes LoRA matrices directly
    onto the Linear layers that expose lora_A / lora_B buffers.

    Non-sticky: the composition is re-evaluated every forward pass so it
    correctly tracks ComfyUI's cloned model patchers.
    """

    def __init__(self):
        self.current_lora_id = None

    def pre_forward(self, module, input_args, input_kwargs):
        transformer_options = input_kwargs.get("transformer_options", {})
        if not transformer_options:
            ctx = input_args[2] if len(input_args) > 2 else None
            if isinstance(ctx, dict):
                transformer_options = ctx.get("transformer_options", {})

        dynamic_loras = transformer_options.get("dynamic_loras", [])
        lora_id = (
            hash(tuple((id(d["patches"]), d["strength"]) for d in dynamic_loras))
            if dynamic_loras else None
        )

        if lora_id == self.current_lora_id:
            return None

        self._apply_composition(module, dynamic_loras)
        self.current_lora_id = lora_id
        return None

    def _apply_composition(self, diffusion_model, dynamic_loras):
        # Pre-group patches by layer key
        layer_patches: dict = {}
        for entry in dynamic_loras:
            strength = entry["strength"]
            for key, adapter in entry["patches"].items():
                layer_patches.setdefault(key, []).append((adapter, strength))

        for name, module in diffusion_model.named_modules():
            if not hasattr(module, "lora_A"):
                continue

            possible_keys = [
                f"diffusion_model.{name}.weight",
                f"{name}.weight",
            ]
            patches = None
            for pk in possible_keys:
                if pk in layer_patches:
                    patches = layer_patches[pk]
                    break

            if not patches:
                module.lora_A = None
                module.lora_B = None
                module.lora_alpha = None
                continue

            all_A, all_B = [], []
            for adapter, strength in patches:
                v = adapter.weights
                up, down, alpha = v[0], v[1], v[2]
                mid = v[3] if len(v) > 3 else None
                rank = down.shape[0] if down.ndim >= 2 else 1
                scale = (alpha / rank) * strength if alpha is not None else strength

                curr_A = down
                if mid is not None:
                    curr_A = torch.mm(mid.flatten(1), down.flatten(1)).reshape(down.shape)

                all_A.append(curr_A * scale)
                all_B.append(up)

            if all_A:
                dev = getattr(module, "weight", torch.tensor(0)).device
                module.lora_A = torch.cat(all_A, dim=0).to(dev)
                module.lora_B = torch.cat(all_B, dim=1).to(dev)
                module.lora_alpha = None
            else:
                module.lora_A = None
                module.lora_B = None

    @classmethod
    def register(cls, diffusion_model):
        """Idempotent — safe to call multiple times on the same model."""
        if not hasattr(diffusion_model, "_wint8_dynamic_lora_hook"):
            hook = cls()
            diffusion_model._wint8_dynamic_lora_hook = hook
            diffusion_model.register_forward_pre_hook(
                hook.pre_forward, with_kwargs=True
            )
        return diffusion_model._wint8_dynamic_lora_hook


# =============================================================================
# Layout registration
# =============================================================================

def _register_layouts():
    try:
        from comfy.quant_ops import QUANT_ALGOS, register_layout_class, QuantizedLayout

        class Int8TensorwiseLayout(QuantizedLayout):
            class Params:
                def __init__(self, scale=None, orig_dtype=None, orig_shape=None, **kwargs):
                    self.scale = scale
                    self.orig_dtype = orig_dtype
                    self.orig_shape = orig_shape

                def clone(self):
                    return Int8TensorwiseLayout.Params(
                        scale=self.scale.clone() if isinstance(self.scale, torch.Tensor) else self.scale,
                        orig_dtype=self.orig_dtype,
                        orig_shape=self.orig_shape,
                    )

            @classmethod
            def state_dict_tensors(cls, qdata, params):
                return {"": qdata, "weight_scale": params.scale}

            @classmethod
            def dequantize(cls, qdata, params):
                return qdata.float() * params.scale

        register_layout_class("Int8TensorwiseLayout", Int8TensorwiseLayout)
        QUANT_ALGOS.setdefault("int8_tensorwise", {
            "storage_t": torch.int8,
            "parameters": {"weight_scale", "input_scale"},
            "comfy_tensor_layout": "Int8TensorwiseLayout",
        })
    except ImportError:
        log.warning("WINT8: ComfyUI quantization system not found (update ComfyUI?)")
    except Exception as e:
        log.error(f"WINT8: Failed to register layouts: {e}")


_register_layouts()


# =============================================================================
# Int8TensorwiseOps — ComfyUI custom operations
# =============================================================================

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False
    log.error("WINT8: comfy.ops not found — cannot define Int8TensorwiseOps")


if _COMFY_OPS_AVAILABLE:
    class Int8TensorwiseOps(manual_cast):
        """
        ComfyUI custom operations for INT8 tensorwise quantization.
        Drop-in replacement for manual_cast; handles INT8 weight loading
        and fast W8A8 forward passes.
        """

        excluded_names      = []
        dynamic_quantize    = False
        enable_quarot       = False
        use_triton          = True
        _is_prequantized    = False

        class Linear(manual_cast.Linear):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer("weight_scale", None)
                self._is_quantized       = False
                self._is_per_row         = False
                self._use_quarot         = False
                self._weight_scale_scalar = None
                self.compute_dtype       = torch.bfloat16
                # Dynamic LoRA slots
                self.lora_A  = None
                self.lora_B  = None
                self.lora_alpha = None

            def reset_parameters(self):
                return None

            def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            ):
                weight_key       = prefix + "weight"
                scale_key        = prefix + "weight_scale"
                input_scale_key  = prefix + "input_scale"
                bias_key         = prefix + "bias"

                weight_scale  = state_dict.pop(scale_key, None)
                weight_tensor = state_dict.pop(weight_key, None)
                state_dict.pop(prefix + "comfy_quant", None)
                state_dict.pop(input_scale_key, None)   # acknowledged, ignored

                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # ── Pre-quantized INT8 checkpoint ─────────────────────
                        self._is_quantized = True
                        Int8TensorwiseOps._is_prequantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)

                        if isinstance(weight_scale, torch.Tensor):
                            if weight_scale.numel() == 1:
                                self._weight_scale_scalar = weight_scale.float().item()
                                self.weight_scale = None
                                self._is_per_row  = False
                            elif weight_scale.dim() == 2 and weight_scale.shape[1] == 1:
                                self.register_buffer("weight_scale", weight_scale.float())
                                self._weight_scale_scalar = None
                                self._is_per_row  = True
                            else:
                                self.register_buffer("weight_scale", weight_scale.float())
                                self._weight_scale_scalar = None
                                self._is_per_row  = False
                        else:
                            self._weight_scale_scalar = float(weight_scale)
                            self.weight_scale = None
                            self._is_per_row  = False

                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        # ── FP checkpoint — optionally quantize on-the-fly ────
                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_dim1 = (
                            self.in_features == 1
                            or self.out_features == 1
                            or weight_tensor.ndim == 1
                        )

                        if is_excluded or is_dim1 or not Int8TensorwiseOps.dynamic_quantize:
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        else:
                            # On-the-fly quantization
                            device = torch.device("cuda") if torch.cuda.is_available() else weight_tensor.device
                            w_gpu  = weight_tensor.to(device).float()

                            self._use_quarot = False
                            if (
                                getattr(Int8TensorwiseOps, "enable_quarot", False)
                                and self.in_features % 128 == 0
                            ):
                                try:
                                    from .wint8_quarot import build_hadamard, rotate_weight
                                    H = build_hadamard(128, device=w_gpu.device, dtype=w_gpu.dtype)
                                    w_gpu = rotate_weight(w_gpu, H, group_size=128)
                                    self._use_quarot = True
                                except Exception as e:
                                    log.warning(f"WINT8: QuaRot failed: {e}")

                            q_weight, q_scale = quantize_int8_axiswise(w_gpu, dim=1)
                            self.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
                            self.register_buffer("weight_scale", q_scale.cpu())
                            self._weight_scale_scalar = None
                            self._is_quantized = True
                            self._is_per_row   = True
                    else:
                        self._is_quantized = False
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    missing_keys.append(weight_key)

                bias_tensor = state_dict.pop(bias_key, None)
                self.bias = nn.Parameter(bias_tensor, requires_grad=False) if bias_tensor is not None else None

                # Keep ComfyUI dtype metadata accurate
                if self.weight is not None:
                    self.weight_comfy_model_dtype = self.weight.dtype
                if self.weight_scale is not None:
                    self.weight_scale_comfy_model_dtype = self.weight_scale.dtype
                if self.bias is not None:
                    self.bias_comfy_model_dtype = self.bias.dtype

            # ── Weight scale accessor ─────────────────────────────────────────

            def _get_weight_scale(self):
                if self._weight_scale_scalar is not None:
                    return self._weight_scale_scalar
                return self.weight_scale

            # ── ComfyUI patching hooks ────────────────────────────────────────

            def convert_weight(self, _weight, inplace=False):
                if not self._is_quantized:
                    return _weight
                return self.weight

            def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if not self._is_quantized:
                    new_w = out_weight.to(self.weight.dtype)
                    if return_weight:
                        return new_w
                    if inplace_update:
                        self.weight.data.copy_(new_w)
                    else:
                        self.weight = nn.Parameter(new_w, requires_grad=False)
                    return

                if out_weight.dtype == torch.int8:
                    if return_weight:
                        return out_weight
                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                # Re-quantize if a float delta slipped through
                new_w = quantize_int8(out_weight, self._get_weight_scale())
                if return_weight:
                    return new_w
                if inplace_update:
                    self.weight.data.copy_(new_w)
                else:
                    self.weight = nn.Parameter(new_w, requires_grad=False)

            def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if out_bias is None:
                    return None
                if return_weight:
                    return out_bias
                if inplace_update and self.bias is not None:
                    self.bias.data.copy_(out_bias)
                else:
                    self.bias = nn.Parameter(out_bias, requires_grad=False)

            # ── Forward ───────────────────────────────────────────────────────

            def forward(self, x: Tensor) -> Tensor:
                need_cast = (
                    self.comfy_cast_weights
                    or len(self.weight_function) > 0
                    or len(self.bias_function) > 0
                )

                if not self._is_quantized:
                    if need_cast:
                        weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                        out = F.linear(x, weight, bias)
                        uncast_bias_weight(self, weight, bias, offload_stream)
                        return out
                    return F.linear(x, self.weight, self.bias)

                # INT8 path
                if need_cast:
                    weight, bias, offload_stream = cast_bias_weight(
                        self, input=None, dtype=torch.int8, device=x.device,
                        bias_dtype=x.dtype, offloadable=True,
                    )
                else:
                    weight, bias, offload_stream = self.weight, self.bias, None

                w_scale = self._get_weight_scale()
                if isinstance(w_scale, torch.Tensor) and w_scale.device != x.device:
                    w_scale = w_scale.to(x.device, non_blocking=True)

                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])

                # Optional QuaRot activation rotation
                if getattr(self, "_use_quarot", False):
                    try:
                        from .wint8_quarot import build_hadamard, rotate_activation
                        H = build_hadamard(128, device=x.device, dtype=x.dtype)
                        x_2d = rotate_activation(x_2d, H, group_size=128)
                    except Exception:
                        pass

                # Sync triton toggle
                import sys
                _mod = sys.modules[__name__]
                _mod._use_triton = Int8TensorwiseOps.use_triton

                if x_2d.shape[0] > 16:
                    if self._is_per_row:
                        y = int8_forward_dynamic_per_row(x_2d, weight, w_scale, bias, compute_dtype)
                    else:
                        y = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
                else:
                    # Small-batch fallback: dequantize + standard linear
                    w_float = dequantize(weight, w_scale).to(x.dtype)
                    bias_t  = bias.to(x.dtype) if bias is not None else None
                    y = F.linear(x_2d, w_float, bias_t)

                # Dynamic LoRA path
                if self.lora_A is not None and self.lora_B is not None:
                    lA = self.lora_A.to(x.device, non_blocking=True)
                    lB = self.lora_B.to(x.device, non_blocking=True)
                    lora_x = F.linear(x_2d.to(lA.dtype), lA)
                    lora_y = F.linear(lora_x, lB)
                    if self.lora_alpha is not None:
                        lora_y = lora_y * self.lora_alpha
                    y = y + lora_y.to(y.dtype)

                if need_cast:
                    uncast_bias_weight(self, weight, bias, offload_stream)

                return y.reshape(*x_shape[:-1], y.shape[-1])

        # ── Pass-through for non-linear layers ────────────────────────────────
        class GroupNorm(manual_cast.GroupNorm):       pass
        class LayerNorm(manual_cast.LayerNorm):       pass
        class Conv2d(manual_cast.Conv2d):             pass
        class Conv3d(manual_cast.Conv3d):             pass
        class ConvTranspose2d(manual_cast.ConvTranspose2d): pass
        class Embedding(manual_cast.Embedding):       pass

        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:   return cls.Conv2d(*args, **kwargs)
            elif dims == 3: return cls.Conv3d(*args, **kwargs)
            raise ValueError(f"WINT8: unsupported conv dims: {dims}")
