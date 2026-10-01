"""Vanilla-PyTorch inference runtime for KhalaModel — the de-Megatron-ified replacement
for Megatron's StaticInferenceEngine (backbone) and generate_superres_manual_projection
(super-res). Device-agnostic via core.device_utils. No Megatron / TransformerEngine.
"""
from __future__ import annotations

import torch
from safetensors.torch import load_file

from .khala_config import KhalaConfig
from .khala_model import KhalaModel, KhalaKVCache
from .device_utils import empty_cache as _empty_cache

# Super-res token-space constants (mirror backend_worker).
PAD_TOKEN_ID = -1
NUM_QUANTIZERS = 64
VQ0_START_ID = 128256
TASK0_MARKER = 128255


def load_vanilla_model(label: str, weights_path, args_json, device, dtype) -> KhalaModel:
    """Build a KhalaModel from converted safetensors + the gathered megatron args."""
    cfg = KhalaConfig.from_megatron_args(str(args_json))
    model = KhalaModel(cfg)
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    model.to(device=device, dtype=dtype).eval()
    return model


def _select_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """logits: [V] (already real-vocab-sliced, fp32). Greedy when temp==0 or top_k==1."""
    if temperature == 0 or top_k == 1:
        return int(logits.argmax())
    logits = logits / max(temperature, 1e-6)
    k = min(int(top_k), logits.shape[-1])
    vals, idx = torch.topk(logits, k)
    probs = torch.softmax(vals, dim=-1)
    g = torch.empty_like(probs).exponential_(1.0)   # gumbel-max top-k (matches super-res)
    sel = int(torch.argmax(probs / g))
    return int(idx[sel])


@torch.inference_mode()
def sample_backbone(model: KhalaModel, prompt_ids, num_tokens: int,
                    temperature: float = 0.0, top_k: int = 1, eos_id: int | None = None):
    """KV-cached autoregressive decode. Returns list[int] of generated token ids."""
    cfg = model.config
    device = next(model.parameters()).device
    V = cfg.vocab_size
    cache = KhalaKVCache(cfg.num_layers)

    seq = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    h = model.forward_hidden_states(seq, causal=True, kv_cache=cache)
    logits = model.lm_head(h[:, -1])[0, :V].float()

    out: list[int] = []
    for step_i in range(int(num_tokens)):
        nxt = _select_token(logits, temperature, top_k)
        if eos_id is not None and nxt == eos_id:
            break  # EOS is a control token, not audio — exclude it from the q0/q1 stream
        out.append(nxt)
        step = torch.tensor([[nxt]], dtype=torch.long, device=device)
        h = model.forward_hidden_states(step, causal=True, kv_cache=cache)
        logits = model.lm_head(h[:, -1])[0, :V].float()
        # The KV cache grows by one each step and attention re-materializes the
        # GQA-expanded K/V at the new length. On MPS the caching allocator hoards every
        # freed (ever-larger) buffer -> ~O(S^2) cached memory over a long decode (e.g.
        # ~73GB at 7k tokens). Periodically return it so memory stays bounded.
        if (step_i & 63) == 63:
            _empty_cache(device)
    return out


@torch.inference_mode()
def generate_superres_projection(model: KhalaModel, tokens: torch.Tensor,
                                 attention_mask: torch.Tensor, loss_mask: torch.Tensor,
                                 position_ids: torch.Tensor, text_len: int, audio_len: int,
                                 top_k: int) -> torch.Tensor:
    """Expand q0/q1 -> q0..q63 by projecting non-causal hidden states onto each
    quantizer's 1024-token vocab window. Vanilla port of
    backend_worker.generate_superres_manual_projection (which returned seq-first hidden
    and transposed; our forward_hidden_states already returns [B, S, H], so no transpose).
    Returns [NUM_QUANTIZERS, 1, audio_len] (token-id space, offsets NOT yet subtracted)."""
    final = torch.zeros(1, tokens.size(1), NUM_QUANTIZERS,
                        device=tokens.device, dtype=tokens.dtype)
    final[..., :2] = tokens[..., :2]
    output_weight = model.lm_head.weight
    device = next(model.parameters()).device

    for idx in range(2, NUM_QUANTIZERS):
        final[:, text_len - 1, 0] = TASK0_MARKER - idx
        hidden = model.forward_hidden_states(
            input_ids=final[..., :idx], causal=False,
            position_ids=position_ids, attention_mask=attention_mask,
        )  # [1, S, H]

        min_id = VQ0_START_ID + idx * 1024
        max_id = min_id + 1024
        ow = output_weight[min_id:max_id].to(hidden.dtype)
        logits = torch.matmul(hidden, ow.t()).float()   # [1, S, 1024]

        k = min(max(1, int(top_k)), logits.shape[-1])
        top_values, top_indices = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(top_values.float(), dim=-1)
        q = torch.empty_like(probs).exponential_(1.0)
        sampled_in_topk = torch.argmax(probs / q, dim=-1)
        sampled = torch.gather(top_indices, -1, sampled_in_topk.unsqueeze(-1)).squeeze(-1) + min_id
        sampled = torch.where(loss_mask.bool(), sampled, PAD_TOKEN_ID)
        final[..., idx] = sampled

        # Release the per-quantizer attention/projection tensors before the next iteration.
        # On MPS the caching allocator otherwise holds the O(S^2) attention high-water mark
        # across all 62 passes; empty_cache returns it so memory stays bounded.
        del hidden, logits, top_values, top_indices, probs, q, sampled_in_topk, sampled
        _empty_cache(device)

    audio = final[:, text_len:text_len + audio_len, :]
    return audio.permute(2, 0, 1).clone()
