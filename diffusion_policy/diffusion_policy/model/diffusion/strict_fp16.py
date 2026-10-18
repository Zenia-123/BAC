"""Inference-only binary16 decoder arithmetic, without hidden FP32 reductions.

CUDA kernels deliberately use separate PTX mul.rn.f16 / add.rn.f16. Unary
functions use complete binary16 lookup tables prepared on the CPU. Integer
address arithmetic is not a floating-point accumulator. See docs/STRICT_FP16_DECODER.md.
Imported lazily: ordinary FP32 inference does not require Triton.
"""
import json
import re
import subprocess
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
from torch import nn
import triton
import triton.language as tl

_COMPILED = {}


def _launch(kernel, grid, *args, **kwargs):
    compiled = kernel[grid](*args, **kwargs, enable_fp_fusion=False)
    _COMPILED[(compiled.name, compiled.hash)] = compiled


@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise("add.rn.f16 $0, $1, $2;", constraints="=h,h,h",
                                    args=[a, b], dtype=tl.float16, is_pure=True, pack=1)


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise("mul.rn.f16 $0, $1, $2;", constraints="=h,h,h",
                                    args=[a, b], dtype=tl.float16, is_pure=True, pack=1)


@triton.jit
def _neg(a):
    return (a.to(tl.uint16, bitcast=True) ^ 32768).to(tl.float16, bitcast=True)


@triton.jit
def _lut(x, table):
    return tl.load(table + x.to(tl.uint16, bitcast=True).to(tl.int32))


@triton.jit
def _mm(A, B, Bias, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        ab: tl.constexpr, am: tl.constexpr, ak: tl.constexpr,
        bb: tl.constexpr, bk: tl.constexpr, bn: tl.constexpr,
        HAS_BIAS: tl.constexpr, R: tl.constexpr):
    # One row and eight output columns per CTA; pairwise half reduction over K.
    batch = tl.program_id(0)
    row = tl.program_id(1)
    cols = tl.program_id(2) * 8 + tl.arange(0, 8)
    ks = tl.arange(0, R)
    av = tl.load(A + batch * ab + row * am + ks * ak, ks < K, other=0)
    bv = tl.load(B + batch * bb + ks[None, :] * bk + cols[:, None] * bn,
                 (ks[None, :] < K) & (cols[:, None] < N), other=0)
    prod = _mul(av[None, :], bv)
    acc = tl.reduce(prod, 1, _add)
    if HAS_BIAS:
        acc = _add(acc, tl.load(Bias + cols, cols < N, other=0))
    tl.store(Out + (batch * M + row) * N + cols, acc, cols < N)


@triton.jit
def _element(A, B, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Out + i, _add(tl.load(A+i, i<N, other=0), tl.load(B+i, i<N, other=0)), i<N)


@triton.jit
def _unary(A, Table, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(A+i, i<N, other=0)
    tl.store(Out+i, _lut(v, Table), i<N)


@triton.jit
def _norm(A, W, Bias, Rsqrt, Out, D: tl.constexpr, R: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, R)
    x = tl.load(A + row*D + i, i<D, other=0)
    scale = tl.full((), 1.0 / D, tl.float16)
    mean = _mul(tl.reduce(x, 0, _add), scale)
    centered = _add(x, _neg(mean))
    square = tl.where(i<D, _mul(centered, centered), tl.full((), 0, tl.float16))
    variance = _mul(tl.reduce(square, 0, _add), scale)
    inv = _lut(_add(variance, tl.full((), EPS, tl.float16)), Rsqrt)
    y = _add(_mul(_mul(centered, inv), tl.load(W+i, i<D, other=0)),
             tl.load(Bias+i, i<D, other=0))
    tl.store(Out + row*D + i, y, i<D)


@triton.jit
def _softmax(A, Mask, Exp, Reciprocal, Out, S: tl.constexpr, T: tl.constexpr,
             R: tl.constexpr, SCALE: tl.constexpr, HAS_MASK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, R)
    x = _mul(tl.load(A + row*S+j, j<S, other=0), tl.full((), SCALE, tl.float16))
    if HAS_MASK:
        x = _add(x, tl.load(Mask + (row % T)*S+j, j<S, other=0))
    x = tl.where(j<S, x, tl.full((), float('-inf'), tl.float16))
    # Order binary16 bit patterns with integer operations: no FP32 max/compare.
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    keys = tl.where((bits & 32768) != 0, (~bits) & 65535, bits | 32768)
    maxkey = tl.max(keys, 0)
    maxbits = tl.where((maxkey & 32768) != 0, maxkey & 32767, (~maxkey) & 65535)
    maximum = maxbits.to(tl.uint16).to(tl.float16, bitcast=True)
    e = _lut(_add(x, _neg(maximum)), Exp)
    total = tl.reduce(e, 0, _add)
    y = _mul(e, _lut(total, Reciprocal))
    tl.store(Out + row*S+j, y, j<S)


@lru_cache(maxsize=1)
def _cpu_tables():
    # Preparation is outside decoder execution. Every possible half bit pattern
    # has one half result, including signed zero, +/-inf and NaN.
    x = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float64)
    import scipy.special
    with np.errstate(all='ignore'):
        values = {'exp': np.exp(x), 'reciprocal': 1.0/x,
                  'rsqrt': 1.0/np.sqrt(x), 'gelu': x * scipy.special.ndtr(x)}
        values['gelu'][np.isneginf(x)] = -0.0
        return {name: torch.from_numpy(val.astype(np.float16)) for name, val in values.items()}


@lru_cache(maxsize=8)
def _tables(device_index):
    return {name: value.to(torch.device('cuda', device_index))
            for name, value in _cpu_tables().items()}


def _check(x):
    if x.device.type != 'cuda' or x.dtype != torch.float16:
        raise TypeError(f'Strict decoder requires CUDA FP16 tensors, got {x.device}/{x.dtype}')
    if torch.is_grad_enabled():
        raise RuntimeError('Strict decoder is inference-only; use torch.no_grad() or inference_mode().')
    if torch.is_autocast_enabled('cuda'):
        raise RuntimeError('Disable autocast for strict FP16 inference.')


def matmul(a, b, bias=None):
    """[batch,M,K] @ [batch,K,N], B may have batch size 1."""
    _check(a)
    _check(b)
    if a.ndim != 3 or b.ndim != 3 or a.shape[-1] != b.shape[-2]:
        raise ValueError('Strict matmul expects compatible rank-three inputs.')
    batch, m, k = a.shape
    if b.shape[0] not in (1, batch):
        raise ValueError('Incompatible matmul batch dimensions.')
    n = b.shape[-1]
    out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
    _launch(_mm, (batch, m, triton.cdiv(n, 8)), a, b, bias if bias is not None else a, out,
            m, n, k, *a.stride(), b.stride(0) if b.shape[0] != 1 else 0,
            b.stride(1), b.stride(2), bias is not None, triton.next_power_of_2(k), num_warps=4)
    return out


def linear(x, weight, bias):
    shape = x.shape[:-1] + (weight.shape[0],)
    return matmul(x.reshape(1, -1, x.shape[-1]), weight.t().unsqueeze(0), bias).reshape(shape)


def residual_add(a, b):
    _check(a)
    _check(b)
    if a.shape != b.shape:
        raise ValueError('Residual shapes must match.')
    a, b = a.contiguous(), b.contiguous()
    out = torch.empty_like(a)
    _launch(_element, (triton.cdiv(a.numel(), 256),), a, b, out, a.numel(), 256)
    return out


class StrictLinear(nn.Linear):
    def forward(self, x):
        return linear(x, self.weight, self.bias)


class StrictLayerNorm(nn.LayerNorm):
    def forward(self, x):
        _check(x)
        x = x.contiguous()
        d = x.shape[-1]
        out = torch.empty_like(x)
        _launch(_norm, (x.numel() // d,), x, self.weight, self.bias,
                _tables(x.device.index)['rsqrt'], out, d, triton.next_power_of_2(d), self.eps)
        return out


def gelu(x):
    _check(x)
    x = x.contiguous()
    out = torch.empty_like(x)
    _launch(_unary, (triton.cdiv(x.numel(), 256),), x, _tables(x.device.index)['gelu'],
            out, x.numel(), 256)
    return out


class StrictAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if self.training or key_padding_mask is not None or need_weights or not self.batch_first:
            raise ValueError('Strict attention supports eval, batch_first, no padding, need_weights=False.')
        if is_causal and attn_mask is None:
            raise ValueError('Strict attention requires an explicit causal mask.')
        b, t, d = query.shape
        s = key.shape[1]
        h, dh = self.num_heads, d // self.num_heads
        w = self.in_proj_weight
        bias = self.in_proj_bias
        q, k, v = [linear(x, w[i*d:(i+1)*d], None if bias is None else bias[i*d:(i+1)*d])
                   for i, x in enumerate((query, key, value))]
        q = q.reshape(b,t,h,dh).transpose(1,2).reshape(b*h,t,dh)
        k = k.reshape(b,s,h,dh).transpose(1,2).reshape(b*h,s,dh)
        v = v.reshape(b,s,h,dh).transpose(1,2).reshape(b*h,s,dh)
        scores = matmul(q, k.transpose(1,2))
        p = torch.empty_like(scores)
        if attn_mask is not None:
            _check(attn_mask)
            if attn_mask.shape != (t,s):
                raise ValueError('Only two-dimensional additive attention masks are supported.')
            attn_mask = attn_mask.contiguous()
        _launch(_softmax, (b*h*t,), scores, attn_mask if attn_mask is not None else scores,
                _tables(query.device.index)['exp'], _tables(query.device.index)['reciprocal'],
                p, s, t, triton.next_power_of_2(s), dh**-0.5, attn_mask is not None)
        x = matmul(p, v).reshape(b,h,t,dh).transpose(1,2).reshape(b,t,d)
        return self.out_proj(x), None


class StrictDecoderLayer(nn.TransformerDecoderLayer):
    _residual_add = staticmethod(residual_add)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        if self.training or not self.norm_first:
            raise ValueError('Strict decoder supports eval and norm_first=True only.')
        x = residual_add(tgt, self._sa_block(self.norm1(tgt), tgt_mask, tgt_key_padding_mask, tgt_is_causal))
        x = residual_add(x, self._mha_block(self.norm2(x), memory, memory_mask,
                                           memory_key_padding_mask, memory_is_causal))
        return residual_add(x, self._ff_block(self.norm3(x)))


class StrictDecoder(nn.TransformerDecoder):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=False):
        # The model already supplies explicit additive masks. Avoid the native
        # container's automatically generated comparison-mask CUDA kernels.
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      tgt_is_causal=bool(tgt_is_causal), memory_is_causal=memory_is_causal)
        return self.norm(x) if self.norm is not None else x


def configure(model, precision):
    if precision != 'decoder_fp16_strict':
        raise ValueError(f'Unknown precision: {precision}')
    if model.training or model.device.type != 'cuda' or model.encoder_only:
        raise ValueError('Strict precision requires an eval-mode CUDA encoder-decoder lowdim Transformer.')
    if getattr(model, 'inference_precision', 'fp32') == precision:
        return model
    # Validate before changing any parameter dtype or module implementation.
    for module in model.decoder.modules():
        if isinstance(module, nn.TransformerDecoderLayer):
            if not module.norm_first or module.activation is not torch.nn.functional.gelu:
                raise ValueError('Strict decoder requires norm_first=True and GELU activation.')
        elif isinstance(module, nn.MultiheadAttention):
            if (not module.batch_first or not module._qkv_same_embed_dim
                    or module.bias_k is not None or module.bias_v is not None or module.add_zero_attn):
                raise ValueError('Unsupported strict attention configuration.')
        elif isinstance(module, nn.LayerNorm):
            if not module.elementwise_affine or module.bias is None:
                raise ValueError('Strict LayerNorm requires affine weight and bias.')
    _tables(model.device.index)  # Precompute/transfer before forward or timing.
    for module in model.decoder.modules():
        if isinstance(module, nn.TransformerDecoder):
            module.__class__ = StrictDecoder
        elif isinstance(module, nn.TransformerDecoderLayer):
            module.__class__ = StrictDecoderLayer
            module._inference_precision = precision
            module.activation = gelu
        elif isinstance(module, nn.MultiheadAttention):
            module.__class__ = StrictAttention
        elif isinstance(module, nn.Linear):
            module.__class__ = StrictLinear
        elif isinstance(module, nn.LayerNorm):
            module.__class__ = StrictLayerNorm
    model.ln_f.__class__ = StrictLayerNorm
    model.decoder.half()
    model.ln_f.half()
    model.inference_precision = precision
    return model


def export_kernel_audit(output_dir):
    """Export executed specializations and disassembly; fail closed on FP32 math."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    bindir = Path(triton.__file__).parent / 'backends/nvidia/bin'
    records = []
    for (name, digest), compiled in _COMPILED.items():
        stem = root / f'{name}_{digest[:12]}'
        ptx = compiled.asm['ptx']
        stem.with_suffix('.ptx').write_text(ptx)
        stem.with_suffix('.cubin').write_bytes(compiled.asm['cubin'])
        sass = subprocess.check_output([str(bindir/'cuobjdump'), '--dump-sass', str(stem.with_suffix('.cubin'))], text=True)
        stem.with_suffix('.sass').write_text(sass)
        # Do not count .file/.loc/source comments as instructions.
        ptx_instructions = '\n'.join(line for line in ptx.splitlines()
                                      if line.strip() and not line.lstrip().startswith(('.', '//')))
        forbidden_ptx = [line.strip() for line in ptx_instructions.splitlines()
                         if re.search(r'\.(?:f32|f64|tf32|bf16)\b|\b(?:mma|wmma)\.', line)]
        forbidden_sass = re.findall(
            r'\b(?:FADD|FMUL|FFMA|FSET\w*|FMNMX|FSEL|FCMP|F2\w*|I2F|'
            r'DADD|DMUL|DFMA|MUFU|HMMA|IMMA|DMMA|WGMMA)\b[^;\n]*', sass)
        records.append({'kernel': name, 'hash': digest, 'ptx': str(stem.with_suffix('.ptx')),
                        'sass': str(stem.with_suffix('.sass')), 'forbidden_ptx': forbidden_ptx,
                        'forbidden_sass': forbidden_sass})
    report = {'precision': 'decoder_fp16_strict', 'kernels': records,
              'passed': bool(records) and not any(r['forbidden_ptx'] or r['forbidden_sass'] for r in records),
              'scope': 'Executed decoder kernels only; encoder/head/scheduler and LUT preparation excluded.'}
    (root / 'audit.json').write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise RuntimeError(f'Strict kernel instruction audit failed: {root}/audit.json')
    return report
