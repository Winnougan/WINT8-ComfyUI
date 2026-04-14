"""
wint8_blockwise.py
──────────────────
Self-contained block-wise INT8 quantization for WINT8 nodes.

Block-wise scaling gives each (block_size × block_size) tile its own scale
factor, giving much finer quantization granularity than tensorwise (one global
scale) — especially valuable for large transformer weight matrices with
localised outliers.

Triton kernels are used when available; falls back to pure PyTorch otherwise.

No dependency on QuantOps, comfy_kitchen, or any other custom node.
"""

import logging
import torch
import torch.nn.functional as F
from torch import Tensor, nn

log = logging.getLogger("WINT8")

# ── Triton availability ───────────────────────────────────────────────────────

_TRITON_AVAILABLE = False
_triton_fns: dict = {}

def _try_load_triton():
    global _TRITON_AVAILABLE, _triton_fns
    try:
        import triton
        import triton.language as tl

        # ── Kernel: activation quantization (per block-row) ───────────────────
        @triton.jit
        def _act_quant_kernel(
            x_ptr, q_ptr, s_ptr,
            N, BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0)
            blk = tl.program_id(1)
            offs = tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(x_ptr + row * N + blk * BLOCK + offs, mask=mask, other=0.0)
            amax = tl.max(tl.abs(x), axis=0)
            scale = tl.maximum(amax / 127.0, 1e-8)
            q = tl.math.round(x / scale)
            q = tl.clamp(q, -128.0, 127.0).to(tl.int8)
            tl.store(q_ptr + row * N + blk * BLOCK + offs, q, mask=mask)
            tl.store(s_ptr + row * (N // BLOCK) + blk, scale.to(tl.float32))

        def triton_act_quant(x: torch.Tensor, block_size: int = 128):
            """Quantize activations [..., K] block-wise. Returns (int8, scale)."""
            orig = x.shape
            K = orig[-1]
            rows = x.numel() // K
            x2 = x.reshape(rows, K).contiguous()
            q = torch.empty_like(x2, dtype=torch.int8)
            s = torch.empty(rows, K // block_size, device=x.device, dtype=torch.float32)
            grid = (rows, K // block_size)
            _act_quant_kernel[grid](x2, q, s, K, BLOCK=block_size)
            return q.reshape(orig), s.reshape(*orig[:-1], K // block_size)

        # ── Kernel: weight quantization (2-D block tiling) ────────────────────
        @triton.jit
        def _weight_quant_kernel(
            w_ptr, q_ptr, s_ptr,
            M, N, BLOCK: tl.constexpr,
        ):
            bm = tl.program_id(0)
            bn = tl.program_id(1)
            offs_m = tl.arange(0, BLOCK)
            offs_n = tl.arange(0, BLOCK)
            ptrs = w_ptr + (bm * BLOCK + offs_m[:, None]) * N + (bn * BLOCK + offs_n[None, :])
            w = tl.load(ptrs)
            amax = tl.max(tl.abs(w))
            scale = tl.maximum(amax / 127.0, 1e-8)
            q = tl.math.round(w / scale)
            q = tl.clamp(q, -128.0, 127.0).to(tl.int8)
            tl.store(
                q_ptr + (bm * BLOCK + offs_m[:, None]) * N + (bn * BLOCK + offs_n[None, :]),
                q,
            )
            tl.store(s_ptr + bm * (N // BLOCK) + bn, scale.to(tl.float32))

        def triton_weight_quant(w: torch.Tensor, block_size: int = 128):
            """Quantize weight (M, N) block-wise. Returns (int8, scale)."""
            M, N = w.shape
            q = torch.empty_like(w, dtype=torch.int8)
            s = torch.empty(M // block_size, N // block_size, device=w.device, dtype=torch.float32)
            grid = (M // block_size, N // block_size)
            _weight_quant_kernel[grid](w.contiguous(), q, s, M, N, BLOCK=block_size)
            return q, s

        # ── Kernel: INT8 GEMM with block-wise dequant epilogue ────────────────
        @triton.autotune(
            configs=[
                triton.Config({'BM': 64,  'BN': 128, 'BK': 64,  'G': 8}, num_stages=3, num_warps=4),
                triton.Config({'BM': 128, 'BN': 128, 'BK': 64,  'G': 8}, num_stages=3, num_warps=8),
                triton.Config({'BM': 64,  'BN': 64,  'BK': 32,  'G': 8}, num_stages=4, num_warps=4),
                triton.Config({'BM': 128, 'BN': 64,  'BK': 32,  'G': 8}, num_stages=4, num_warps=4),
            ],
            key=['M', 'N', 'K'],
        )
        @triton.jit
        def _int8_gemm_blockwise_kernel(
            a_ptr, b_ptr, c_ptr,
            as_ptr, bs_ptr,
            M, N, K,
            sa, sb, sa_s, sb_s,
            sc, sd,
            BLOCK_SIZE: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            G: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            bias_ptr,
        ):
            pid = tl.program_id(0)
            num_m = tl.cdiv(M, BM); num_n = tl.cdiv(N, BN)
            in_g = G * num_n
            gid = pid // in_g
            fm = gid * G
            gsz = tl.minimum(num_m - fm, G)
            pid_m = fm + (pid % gsz)
            pid_n = (pid % in_g) // gsz
            offs_m = (pid_m * BM + tl.arange(0, BM)) % M
            offs_n = (pid_n * BN + tl.arange(0, BN)) % N
            offs_k = tl.arange(0, BK)
            ap = a_ptr + offs_m[:, None] * sa + offs_k[None, :] * 1
            bp = b_ptr + offs_k[:, None] * 1  + offs_n[None, :] * sb
            acc = tl.zeros((BM, BN), dtype=tl.int32)
            for k in range(0, tl.cdiv(K, BK)):
                a = tl.load(ap, mask=offs_k[None, :] < K - k * BK, other=0)
                b = tl.load(bp, mask=offs_k[:, None] < K - k * BK, other=0)
                acc += tl.dot(a, b)
                ap += BK
                bp += BK * sb
            # Scale indices
            sm = offs_m // BLOCK_SIZE
            sn = offs_n // BLOCK_SIZE
            sk = (tl.cdiv(K, BLOCK_SIZE) - 1)   # last k-block index for act scale
            a_sc = tl.load(as_ptr + offs_m * sa_s + sk)          # [BM]
            b_sc = tl.load(bs_ptr + sm[:, None] * sb_s + sn[None, :])  # [BM, BN]  (approx)
            c = acc.to(tl.float32) * a_sc[:, None] * b_sc
            if HAS_BIAS:
                bias = tl.load(bias_ptr + offs_n)
                c += bias[None, :]
            cp = c_ptr + offs_m[:, None] * sc + offs_n[None, :] * sd
            cmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(cp, c, mask=cmask)

        def triton_int8_gemm_blockwise(
            a: torch.Tensor, a_scale: torch.Tensor,
            b: torch.Tensor, b_scale: torch.Tensor,
            block_size: int, bias=None,
            out_dtype=torch.bfloat16,
        ):
            M = a.shape[0]; K = a.shape[1]; N = b.shape[0]
            out = torch.empty(M, N, device=a.device, dtype=out_dtype)
            grid = lambda meta: (
                triton.cdiv(M, meta['BM']) * triton.cdiv(N, meta['BN']),
            )
            has_bias = bias is not None
            bias_ptr = bias if has_bias else a
            _int8_gemm_blockwise_kernel[grid](
                a, b, out,
                a_scale, b_scale,
                M, N, K,
                a.stride(0), b.stride(1),
                a_scale.stride(0), b_scale.stride(0),
                out.stride(0), out.stride(1),
                BLOCK_SIZE=block_size,
                HAS_BIAS=has_bias, bias_ptr=bias_ptr,
            )
            return out

        _triton_fns['act_quant']    = triton_act_quant
        _triton_fns['weight_quant'] = triton_weight_quant
        _triton_fns['gemm']         = triton_int8_gemm_blockwise
        _TRITON_AVAILABLE = True
        log.info("WINT8: Blockwise Triton kernels loaded.")

    except Exception as e:
        log.info(f"WINT8: Blockwise Triton kernels unavailable ({e}), using PyTorch fallback.")

_try_load_triton()


# =============================================================================
# PyTorch helpers
# =============================================================================

def _pytorch_act_quant(x: Tensor, block_size: int):
    orig = x.shape; K = orig[-1]
    rows = x.numel() // K
    x2 = x.reshape(rows, K)
    nb = K // block_size
    blocks = x2.reshape(rows, nb, block_size)
    amax = blocks.abs().amax(dim=-1)                      # (rows, nb)
    scale = (amax.float() / 127.0).clamp(min=1e-8)
    q = (blocks.float() / scale.unsqueeze(-1)).round_().clamp_(-128, 127).to(torch.int8)
    return q.reshape(orig), scale.reshape(*orig[:-1], nb)


def _pytorch_weight_quant(w: Tensor, block_size: int):
    M, N = w.shape
    bm, bn = M // block_size, N // block_size
    blocks = w.reshape(bm, block_size, bn, block_size).permute(0, 2, 1, 3)
    amax = blocks.abs().amax(dim=(-2, -1))                # (bm, bn)
    scale = (amax.float() / 127.0).clamp(min=1e-8)
    q_blocks = (blocks.float() / scale.unsqueeze(-1).unsqueeze(-1)).round_().clamp_(-128, 127).to(torch.int8)
    q = q_blocks.permute(0, 2, 1, 3).reshape(M, N)
    return q, scale


def _pytorch_dequant_weight(q: Tensor, scale: Tensor, block_size: int, orig_dtype):
    M, N = q.shape
    bm, bn = M // block_size, N // block_size
    blocks = q.reshape(bm, block_size, bn, block_size).permute(0, 2, 1, 3)
    dq = blocks.float() * scale.unsqueeze(-1).unsqueeze(-1)
    return dq.permute(0, 2, 1, 3).reshape(M, N).to(orig_dtype)


def _pytorch_dequant_act(q: Tensor, scale: Tensor, block_size: int, orig_dtype):
    orig = q.shape; K = orig[-1]
    rows = q.numel() // K; nb = K // block_size
    blocks = q.reshape(rows, nb, block_size).float()
    dq = blocks * scale.reshape(rows, nb, 1)
    return dq.reshape(orig).to(orig_dtype)


def _pytorch_gemm_blockwise(
    a: Tensor, a_scale: Tensor,
    b: Tensor, b_scale: Tensor,
    block_size: int, bias=None, out_dtype=torch.bfloat16,
):
    """Dequantize both operands then do a standard matmul."""
    a_dq = _pytorch_dequant_act(a, a_scale, block_size, out_dtype)
    b_dq = _pytorch_dequant_weight(b, b_scale, block_size, out_dtype)
    return F.linear(a_dq, b_dq, bias.to(out_dtype) if bias is not None else None)


# =============================================================================
# Public API — used by Int8BlockwiseOps
# =============================================================================

def blockwise_quantize_weight(w: Tensor, block_size: int = 128):
    """Quantize a weight matrix to INT8 with block-wise scaling."""
    if _TRITON_AVAILABLE and w.is_cuda:
        return _triton_fns['weight_quant'](w, block_size)
    return _pytorch_weight_quant(w, block_size)


def blockwise_quantize_act(x: Tensor, block_size: int = 128):
    """Quantize activations to INT8 with block-wise scaling."""
    if _TRITON_AVAILABLE and x.is_cuda:
        return _triton_fns['act_quant'](x, block_size)
    return _pytorch_act_quant(x, block_size)


def blockwise_linear(
    x: Tensor, w_int8: Tensor, w_scale: Tensor,
    block_size: int, bias=None, compute_dtype=torch.bfloat16,
):
    """W8A8 linear with block-wise scaling."""
    orig = x.shape
    x2 = x.reshape(-1, orig[-1])

    a_int8, a_scale = blockwise_quantize_act(x2, block_size)

    if _TRITON_AVAILABLE and x2.is_cuda:
        try:
            out = _triton_fns['gemm'](
                a_int8, a_scale, w_int8, w_scale,
                block_size, bias, compute_dtype,
            )
            return out.reshape(*orig[:-1], w_int8.shape[0])
        except Exception as e:
            log.warning(f"WINT8: Blockwise Triton gemm failed ({e}), using PyTorch fallback.")

    out = _pytorch_gemm_blockwise(a_int8, a_scale, w_int8, w_scale, block_size, bias, compute_dtype)
    return out.reshape(*orig[:-1], w_int8.shape[0])


# =============================================================================
# Int8BlockwiseOps — ComfyUI custom operations (blockwise variant)
# =============================================================================

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False


if _COMFY_OPS_AVAILABLE:

    class Int8BlockwiseOps(manual_cast):
        """
        ComfyUI custom operations for block-wise INT8 quantization.
        One scale per (block_size × block_size) tile — finer grained than tensorwise.
        """

        excluded_names   = []
        dynamic_quantize = False
        enable_quarot    = False
        use_triton       = True
        block_size       = 128
        _is_prequantized = False

        class Linear(manual_cast.Linear):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer("weight_scale", None)
                self._is_quantized  = False
                self._use_quarot    = False
                self._block_size    = 128
                self.compute_dtype  = torch.bfloat16
                self.lora_A  = None
                self.lora_B  = None
                self.lora_alpha = None

            def reset_parameters(self):
                return None

            def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            ):
                weight_key  = prefix + "weight"
                scale_key   = prefix + "weight_scale"
                bias_key    = prefix + "bias"

                weight_scale  = state_dict.pop(scale_key, None)
                weight_tensor = state_dict.pop(weight_key, None)
                state_dict.pop(prefix + "comfy_quant",    None)
                state_dict.pop(prefix + "input_scale",    None)

                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # Pre-quantized blockwise checkpoint
                        self._is_quantized = True
                        Int8BlockwiseOps._is_prequantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        ws = weight_scale.float()
                        # Infer block size from scale shape vs weight shape
                        if ws.dim() == 2:
                            self._block_size = weight_tensor.shape[0] // ws.shape[0]
                        else:
                            self._block_size = Int8BlockwiseOps.block_size
                        self.register_buffer("weight_scale", ws)

                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        is_excluded = any(ex in prefix for ex in Int8BlockwiseOps.excluded_names)
                        M, N = weight_tensor.shape[0], (weight_tensor.shape[1] if weight_tensor.ndim > 1 else 1)
                        bs = Int8BlockwiseOps.block_size
                        not_divisible = (M % bs != 0 or N % bs != 0)

                        if is_excluded or weight_tensor.ndim < 2 or not_divisible or not Int8BlockwiseOps.dynamic_quantize:
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        else:
                            device = torch.device("cuda") if torch.cuda.is_available() else weight_tensor.device
                            w_gpu = weight_tensor.to(device).float()

                            self._use_quarot = False
                            if getattr(Int8BlockwiseOps, "enable_quarot", False) and N % 128 == 0:
                                try:
                                    from .wint8_quarot import build_hadamard, rotate_weight
                                    H = build_hadamard(128, device=w_gpu.device, dtype=w_gpu.dtype)
                                    w_gpu = rotate_weight(w_gpu, H, group_size=128)
                                    self._use_quarot = True
                                except Exception as e:
                                    log.warning(f"WINT8: QuaRot failed: {e}")

                            q, scale = blockwise_quantize_weight(w_gpu, bs)
                            self.weight = nn.Parameter(q.cpu(), requires_grad=False)
                            self.register_buffer("weight_scale", scale.cpu())
                            self._block_size    = bs
                            self._is_quantized  = True
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

            def convert_weight(self, _weight, inplace=False):
                return self.weight if self._is_quantized else _weight

            def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if not self._is_quantized or out_weight.dtype == torch.int8:
                    new_w = out_weight if out_weight.dtype == torch.int8 else out_weight.to(self.weight.dtype)
                    if return_weight: return new_w
                    if inplace_update: self.weight.data.copy_(new_w)
                    else: self.weight = nn.Parameter(new_w, requires_grad=False)
                    return
                # Re-quantize float delta
                q, _ = blockwise_quantize_weight(out_weight.float(), self._block_size)
                if return_weight: return q
                if inplace_update: self.weight.data.copy_(q)
                else: self.weight = nn.Parameter(q, requires_grad=False)

            def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if out_bias is None: return None
                if return_weight: return out_bias
                if inplace_update and self.bias is not None:
                    self.bias.data.copy_(out_bias)
                else:
                    self.bias = nn.Parameter(out_bias, requires_grad=False)

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

                if need_cast:
                    weight, bias, offload_stream = cast_bias_weight(
                        self, input=None, dtype=torch.int8, device=x.device,
                        bias_dtype=x.dtype, offloadable=True,
                    )
                else:
                    weight, bias, offload_stream = self.weight, self.bias, None

                w_scale = self.weight_scale
                if w_scale is not None and w_scale.device != x.device:
                    w_scale = w_scale.to(x.device, non_blocking=True)

                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

                x_shape = x.shape
                x2 = x.reshape(-1, x_shape[-1])

                if getattr(self, "_use_quarot", False):
                    try:
                        from .wint8_quarot import build_hadamard, rotate_activation
                        H = build_hadamard(128, device=x.device, dtype=x.dtype)
                        x2 = rotate_activation(x2, H, group_size=128)
                    except Exception:
                        pass

                K = x2.shape[-1]
                bs = self._block_size
                if x2.shape[0] > 16 and K % bs == 0:
                    y = blockwise_linear(x2, weight, w_scale, bs, bias, compute_dtype)
                else:
                    # Small batch or misaligned dims — dequantize
                    w_dq = _pytorch_dequant_weight(weight, w_scale, bs, x.dtype)
                    y = F.linear(x2, w_dq, bias.to(x.dtype) if bias is not None else None)

                # Dynamic LoRA
                if self.lora_A is not None and self.lora_B is not None:
                    lA = self.lora_A.to(x.device, non_blocking=True)
                    lB = self.lora_B.to(x.device, non_blocking=True)
                    lora_y = F.linear(F.linear(x2.to(lA.dtype), lA), lB)
                    if self.lora_alpha is not None:
                        lora_y = lora_y * self.lora_alpha
                    y = y + lora_y.to(y.dtype)

                if need_cast:
                    uncast_bias_weight(self, weight, bias, offload_stream)

                return y.reshape(*x_shape[:-1], y.shape[-1])

        class GroupNorm(manual_cast.GroupNorm):             pass
        class LayerNorm(manual_cast.LayerNorm):             pass
        class Conv2d(manual_cast.Conv2d):                   pass
        class Conv3d(manual_cast.Conv3d):                   pass
        class ConvTranspose2d(manual_cast.ConvTranspose2d): pass
        class Embedding(manual_cast.Embedding):             pass

        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:   return cls.Conv2d(*args, **kwargs)
            elif dims == 3: return cls.Conv3d(*args, **kwargs)
            raise ValueError(f"WINT8: unsupported conv dims: {dims}")
