"""Multi-codebook token embedding, ported from the upstream
`MultiLayerVocabParallelEmbedding` (core/core_dynamic_embedding.py) to vanilla
PyTorch.

The upstream class subclasses Megatron's tensor-parallel `VocabParallelEmbedding`
and carries all the reduce/shard plumbing. With tensor-model-parallel-size == 1
(confirmed for both Khala checkpoints) every tensor-parallel branch collapses to a
no-op, so this is the TP=1 reduction of that logic over a plain `nn.Embedding`.

Three input shapes, matching the upstream task ids:
  * `[B, S]`        standard causal-LM token ids (backbone)            -> [B, S, H]
  * `[B, S, 1]`     q0 -> q1 super-resolution single-codebook          -> [B, S, H]
  * `[B, S, C]`     multi-codebook super-resolution (C = active RVQ)   -> [B, S, H]

Padding convention (verbatim from upstream): `_PAD_TOKEN_ID == -1`; padded
positions are looked up via `NO_USE_TOKEN_ID` then zeroed out. For the `[B, S, C]`
case, column 0 is the text / q0 stream and columns 1.. are audio codebooks; a
position is "audio" when any audio column is non-pad, in which case the text stream
is suppressed and the audio embeddings are summed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .khala_config import KhalaConfig


class MultiLayerEmbedding(nn.Module):
    def __init__(self, config: KhalaConfig):
        super().__init__()
        self.config = config
        self.pad_id = config.pad_token_id           # -1
        self.no_use_id = config.no_use_token_id      # 128004
        # padded_vocab_size rows so it matches the gathered weight matrix exactly.
        self.weight = nn.Parameter(
            torch.empty(config.padded_vocab_size, config.hidden_size)
        )
        # Initialize so a fresh (un-checkpointed) model has finite embeddings;
        # torch.empty leaves uninitialized garbage (occasionally NaN) that would
        # otherwise poison forward passes for randomly-sampled token ids.
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

    @property
    def vocab_size(self) -> int:
        return self.weight.shape[0]

    def _lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        pad_mask = input_ == self.pad_id
        masked = input_.clone()
        masked[pad_mask] = self.no_use_id

        # task 0: [B, S] standard causal LM
        if input_.dim() == 2:
            out = self._lookup(masked)
            return out * (~pad_mask).unsqueeze(-1)

        # task 1: [B, S, 1] q0 -> q1
        if input_.dim() == 3 and input_.size(-1) == 1:
            masked = masked.squeeze(-1)
            pad_mask = pad_mask.squeeze(-1)
            out = self._lookup(masked)
            return out * (~pad_mask).unsqueeze(-1)

        # task >1: [B, S, C] multi-codebook
        if input_.dim() == 3 and input_.size(-1) > 1:
            audio_mask = (input_[..., 1:] != self.pad_id).any(dim=2)  # [B, S]

            # text / q0 stream (column 0); suppressed wherever this position is audio or pad
            text_ids = input_[..., 0]
            text_pad = text_ids == self.pad_id
            text_lookup = text_ids.masked_fill(audio_mask | text_pad, self.no_use_id)
            text_emb = self._lookup(text_lookup)
            text_keep = (~audio_mask) & (~text_pad)
            text_emb = text_emb * text_keep.unsqueeze(-1)

            # audio stream: embed every column, zero pads, sum across codebooks
            masked_all = input_.masked_fill(input_ == self.pad_id, self.no_use_id)
            emb_4d = self._lookup(masked_all)                    # [B, S, C, H]
            nonpad = (input_ != self.pad_id).unsqueeze(-1)
            emb_4d = emb_4d * nonpad
            audio_sum = emb_4d.sum(dim=2) * audio_mask.unsqueeze(-1)  # [B, S, H]

            return text_emb + audio_sum

        raise ValueError(f"Unexpected input dim {input_.dim()}; expected 2 or 3.")
