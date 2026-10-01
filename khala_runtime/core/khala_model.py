"""Vanilla PyTorch KhalaModel — the de-Megatron-ified backbone / super-res transformer.

A standalone `nn.Module` reimplementation of the Megatron-Core GPT used by Khala,
targeting CPU/MPS/CUDA via `core.device_utils`. No Megatron, TransformerEngine,
apex, or flash-attn.

Naming maps to `tools/convert_megatron_to_hf.py` output:
    embed.weight                                  (multi-codebook embedding)
    layers.{i}.input_norm.weight                  pre-attention RMSNorm
    layers.{i}.attn.{q_proj,k_proj,v_proj}.{weight,bias}
    layers.{i}.attn.o_proj.{weight,bias}
    layers.{i}.post_attn_norm.weight              pre-MLP RMSNorm
    layers.{i}.mlp.{gate_proj,up_proj,down_proj}.{weight,bias}
    norm.weight                                   final RMSNorm
    lm_head.weight                                untied output projection

CONVENTIONS NOT YET PARITY-VERIFIED (no CUDA forward goldens captured in this gather):
  * RoPE: implemented as Llama/GPT-NeoX-style rotate_half. Megatron-core's default
    matches this, but the exact half-vs-interleaved split is the #1 source of
    Megatron->HF drift. Flagged; localized to `apply_rotary`.
  * SwiGLU gate/up assignment: assumes silu(gate_proj) * up_proj with
    weight_w -> gate, weight_v -> up. Localized to `KhalaMLP.forward`.
Both are single-point flips once we have a forward fixture to check against.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .khala_config import KhalaConfig
from .khala_embedding import MultiLayerEmbedding
from .device_utils import empty_cache as _empty_cache

# Query-block size for the chunked super-res attention (see _chunked_attention).
_ATTN_BLOCK = int(os.environ.get("KHALA_ATTN_BLOCK", "512"))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Standard RoPE inverse-frequency cache. Returns (cos, sin) of [seq_len, head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)        # [seq_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    # q,k: [B, H, S, D]; cos,sin: [S, D] -> broadcast over B,H
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class KhalaKVCache:
    """Per-layer (k, v) cache for incremental causal decode. k/v are post-RoPE,
    pre-GQA-expansion, shaped [B, n_kv, S, head_dim]."""

    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    def reset(self) -> None:
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None

    def length(self, i: int = 0) -> int:
        return 0 if self.k[i] is None else self.k[i].shape[2]

    def append(self, i: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[i] is None:
            self.k[i], self.v[i] = k, v
        else:
            self.k[i] = torch.cat([self.k[i], k], dim=2)
            self.v[i] = torch.cat([self.v[i], v], dim=2)
        return self.k[i], self.v[i]


def _chunked_attention(q, k, v, attend, block: int):
    """Exact non-causal attention, evaluated as an explicit matmul -> softmax -> matmul
    chunked over the query dimension in fp32. Never materializes the full [B,H,S,S]
    score matrix (peak is [B,H,block,S]) and bypasses the fused MPS SDPA kernel, which
    (a) on the math path materializes O(S^2) scores+probs (~22GB at S=8192 on MPS) and
    (b) silently corrupts results past ~18k tokens (pytorch#179352 / Pixal3D port).

    q,k,v: [B, H, S, head_dim] (k,v already GQA-expanded to H heads).
    attend: [B,1,1,S] bool, True = attend (i.e. ~padding_mask), or None.
    Returns [B, H, S, head_dim] in q's dtype.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    S = q.shape[2]
    out = torch.empty(*q.shape[:-1], v.shape[-1], device=q.device, dtype=q.dtype)
    for s in range(0, S, block):
        e = min(s + block, S)
        scores = (q[:, :, s:e].float() @ k.float().transpose(-2, -1)) * scale  # [B,H,bq,S]
        if attend is not None:
            scores = scores.masked_fill(~attend, float("-inf"))
        out[:, :, s:e] = (scores.softmax(-1) @ v.float()).to(out.dtype)
        del scores
        _empty_cache(q.device)
    return out


class KhalaAttention(nn.Module):
    def __init__(self, config: KhalaConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_query_groups
        self.head_dim = config.head_dim
        bias = config.add_bias_linear
        self.q_proj = nn.Linear(config.hidden_size, config.q_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=bias)
        self.o_proj = nn.Linear(config.q_dim, config.hidden_size, bias=bias)

    def forward(self, x, cos, sin, *, causal: bool = True, attn_mask=None,
                kv_cache=None, layer_idx: int | None = None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)

        q, k = apply_rotary(q, k, cos, sin)

        if kv_cache is not None:
            prior_len = kv_cache.length(layer_idx)
            # Supported: one prefill chunk (prior_len==0, S>=1) then single-token steps (S==1).
            # A multi-token chunk into a non-empty cache would mis-attend under the
            # bottom-right-aligned SDPA causal triangle — reject it rather than fail silently.
            assert not (S > 1 and prior_len > 0), (
                "KhalaKVCache: chunked prefill into a non-empty cache is unsupported "
                f"(S={S}, prior_len={prior_len}); prefill once, then decode one token at a time."
            )
            k, v = kv_cache.append(layer_idx, k, v)  # full cached k/v (post-RoPE)

        # GQA: expand KV heads to match Q heads
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        if attn_mask is not None:
            # Super-res non-causal path. attn_mask [B,1,1,S] (True=pad) -> attend = ~mask.
            # Routed through the chunked matmul+softmax instead of SDPA to bound MPS memory
            # (O(S^2) -> O(block*S), ~22GB -> ~1GB at S=8192) and to dodge the fused MPS
            # kernel's silent long-sequence corruption (pytorch#179352).
            out = _chunked_attention(q, k, v, ~attn_mask, _ATTN_BLOCK)
        elif kv_cache is not None:
            # Prefill chunk (S>1) is causal; a single decode step attends all past keys.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(q.shape[2] > 1))
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class KhalaMLP(nn.Module):
    def __init__(self, config: KhalaConfig):
        super().__init__()
        bias = config.add_bias_linear
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=bias)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=bias)
        self.double_bias = config.swiglu_double_bias and bias

    def forward(self, x):
        # SwiGLU: silu(gate) * up. The upstream fused bias_swiglu adds the fc1 bias a
        # second time (see KhalaConfig.swiglu_double_bias) -> verified bit-exact.
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.double_bias:
            gate = gate + self.gate_proj.bias
            up = up + self.up_proj.bias
        return self.down_proj(F.silu(gate) * up)


class KhalaDecoderLayer(nn.Module):
    def __init__(self, config: KhalaConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = KhalaAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = KhalaMLP(config)

    def forward(self, x, cos, sin, *, causal: bool = True, attn_mask=None,
                kv_cache=None, layer_idx: int | None = None):
        x = x + self.attn(self.input_norm(x), cos, sin, causal=causal,
                          attn_mask=attn_mask, kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class KhalaModel(nn.Module):
    def __init__(self, config: KhalaConfig):
        super().__init__()
        self.config = config
        self.embed = MultiLayerEmbedding(config)
        self.layers = nn.ModuleList(KhalaDecoderLayer(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.padded_vocab_size, bias=False)
        self._rope_cache: tuple | None = None

    def _rope_at(self, positions: torch.Tensor, device, dtype):
        """Return (cos, sin) gathered at absolute `positions` [S] -> each [S, head_dim]."""
        need = int(positions.max().item()) + 1
        c = self._rope_cache
        if (c is None or c[0].shape[0] < need
                or c[0].device != device or c[0].dtype != dtype):
            self._rope_cache = build_rope_cache(
                max(need, 1), self.config.head_dim, self.config.rope_theta, device, dtype
            )
        cos, sin = self._rope_cache
        return cos[positions], sin[positions]

    def forward_hidden_states(self, input_ids: torch.Tensor, *, causal: bool = True,
                              position_ids: torch.Tensor | None = None,
                              attention_mask: torch.Tensor | None = None,
                              kv_cache=None) -> torch.Tensor:
        """Embedding -> transformer stack -> final norm. Returns [B, S, H].

        Defaults (all optional args None/True) reproduce the Phase-1 forward exactly.
        - `causal=False` + `attention_mask` [B,1,1,S] (True=pad): super-res non-causal path.
        - `kv_cache`: incremental causal decode; RoPE positions resume at cache length.
        """
        h = self.embed(input_ids)
        S = h.shape[1]
        start = kv_cache.length() if kv_cache is not None else 0
        if position_ids is None:
            positions = torch.arange(start, start + S, device=h.device)
        else:
            positions = position_ids[0] if position_ids.dim() == 2 else position_ids
        cos, sin = self._rope_at(positions, h.device, h.dtype)
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, causal=causal, attn_mask=attention_mask,
                      kv_cache=kv_cache, layer_idx=i)
        return self.norm(h)

    def forward(self, input_ids: torch.Tensor, *, causal: bool = True,
                position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                kv_cache=None) -> torch.Tensor:
        """Full forward to vocab logits [B, S, padded_vocab_size]."""
        return self.lm_head(self.forward_hidden_states(
            input_ids, causal=causal, position_ids=position_ids,
            attention_mask=attention_mask, kv_cache=kv_cache))
