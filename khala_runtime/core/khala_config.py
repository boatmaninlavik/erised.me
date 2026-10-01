"""Architecture config for the vanilla (de-Megatron-ified) Khala models.

Both the backbone and the super-resolution model share the same transformer
shape (24-layer, 2048-hidden, GQA 32->8, SwiGLU 5632, RMSNorm, RoPE) and differ
only in `seq_length` and vocab size. `from_megatron_args` builds a config straight
from the `*_megatron_args.json` captured during the CUDA gather, so the numbers are
never hand-transcribed.

Recovered from the official checkpoints (iter_0036000 backbone / iter_0010000 superres):
    hidden_size=2048  num_layers=24  heads=32  query_groups=8  head_dim=64
    ffn_hidden_size=5632  swiglu  RMSNorm(eps=1e-6)  RoPE(base=500000)
    add_bias_linear=True  untie_embeddings_and_output_weights=True
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KhalaConfig:
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 32
    num_query_groups: int = 8          # GQA: number of KV heads
    head_dim: int = 64                 # kv_channels
    ffn_hidden_size: int = 5632
    swiglu: bool = True

    norm_eps: float = 1e-6             # norm_epsilon
    rope_theta: float = 500000.0       # rotary_base
    rotary_percent: float = 1.0

    vocab_size: int = 130304           # real vocab
    padded_vocab_size: int = 130432    # weight matrix rows
    max_position_embeddings: int = 16384  # seq_length

    add_bias_linear: bool = True       # every dense layer carries a bias
    untie_embeddings_and_output_weights: bool = True

    # Upstream quirk: the fused bias_swiglu adds the fc1 bias a SECOND time on top
    # of the bias the linear already applied, i.e. silu(W_w x + 2 b_w)*(W_v x + 2 b_v).
    # Verified bit-exact against the reference forward (cos 0.999998). This is very
    # likely the "numerical precision issue affecting inference quality" the upstream
    # README flags. Kept True for parity; set False to get the mathematically-intended
    # single-bias SwiGLU.
    swiglu_double_bias: bool = True

    num_quantizers: int = 64           # RVQ layers (used by the multi-codebook embedding)
    pad_token_id: int = -1             # _PAD_TOKEN_ID in the upstream embedding
    no_use_token_id: int = 128004      # NO_USE_TOKEN_ID in the upstream embedding

    # --- derived ---
    @property
    def kv_dim(self) -> int:
        return self.num_query_groups * self.head_dim   # 8 * 64 = 512

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim  # 32 * 64 = 2048

    @property
    def heads_per_group(self) -> int:
        return self.num_attention_heads // self.num_query_groups  # 4

    def __post_init__(self) -> None:
        assert self.num_attention_heads % self.num_query_groups == 0, (
            "num_attention_heads must be divisible by num_query_groups"
        )
        assert self.q_dim == self.hidden_size, (
            f"q_dim {self.q_dim} != hidden_size {self.hidden_size}; non-square attention "
            "is supported by Megatron but not assumed here — revisit if this trips."
        )

    @classmethod
    def from_megatron_args(cls, args: dict | str | Path) -> "KhalaConfig":
        """Build a KhalaConfig from a gathered `*_megatron_args.json` (dict or path)."""
        if isinstance(args, (str, Path)):
            args = json.loads(Path(args).read_text())

        def g(key: str, default=None):
            return args.get(key, default)

        cfg = cls(
            hidden_size=g("hidden_size"),
            num_layers=g("num_layers"),
            num_attention_heads=g("num_attention_heads"),
            num_query_groups=g("num_query_groups") if g("group_query_attention") else g("num_attention_heads"),
            head_dim=g("kv_channels") or (g("hidden_size") // g("num_attention_heads")),
            ffn_hidden_size=g("ffn_hidden_size"),
            swiglu=bool(g("swiglu", False)),
            norm_eps=float(g("norm_epsilon", 1e-6)),
            rope_theta=float(g("rotary_base", 10000.0)),
            rotary_percent=float(g("rotary_percent", 1.0)),
            vocab_size=g("vocab_size"),
            padded_vocab_size=g("padded_vocab_size"),
            max_position_embeddings=g("max_position_embeddings") or g("seq_length"),
            add_bias_linear=bool(g("add_bias_linear", False)),
            untie_embeddings_and_output_weights=bool(g("untie_embeddings_and_output_weights", True)),
            num_quantizers=g("num_quantizers", 64),
        )
        assert str(g("normalization", "RMSNorm")).lower() == "rmsnorm", (
            f"Expected RMSNorm, got {g('normalization')!r}"
        )
        assert str(g("position_embedding_type", "rope")).lower() in ("rope", "rotary"), (
            f"Expected RoPE, got {g('position_embedding_type')!r}"
        )
        return cfg
