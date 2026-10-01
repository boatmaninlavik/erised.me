"""Shared pieces for DPO post-training of Shao's backbone.

Every dpo_*.py step imports this file, so prompt building, token layout and scoring can't
drift apart between steps. Each constant below was checked against khala_runtime/
(backend_worker.py, khala_model.py, codec yaml) and the backbone's own training args
(gs://erised-khala/mps/backbone_megatron_args.json). test_dpo_common.py checks the logic
on a tiny random model.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---- token layout -------------------------------------------------------------------
VQ0_START_ID = 128256            # first audio id: codec layer-0 code c -> 128256 + c
Q1_OFFSET = 1024                 # codec layer-1 code c -> 128256 + 1024 + c
REAL_VOCAB = 130304              # 128256 text ids + 2 x 1024 audio ids (args: vocab_size)
CONTEXT_LEN = 16384              # backbone seq_length: prompt + audio must fit in this
CODEC_FPS = 44100 / 2048         # 21.53 frames/s (codec strides 4*8*8*8 = 2048 samples)
TOKENS_PER_MINUTE = 2584         # backend_worker constant (= 60 s * 21.53 frames * 2 layers)
SAMPLE_RATE = 44100              # codec input rate; Suno mp3s are 48 kHz -> resample

# ---- prompt special tokens (backend_worker.py) -------------------------------------------
BOS = "<|begin_of_text|>"
EOM = "<|eom_id|>"
BOL = "<|start_header_id|>"
EOT = "<|eot_id|>"
BOA = "<|python_tag|>"
END_OF_LINE = "<|end_header_id|>"

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def build_prompt_text(style: str, lyrics: str, instrumental: bool, duration_min: int) -> str:
    """The exact text Shao's own app builds (backend_worker.compose_prompt_text, tags mode,
    superres_text_mode 'same_as_backbone': the language slot is empty for vocal songs)."""
    if not instrumental:
        lyrics_text = lyrics.strip().replace("\n", END_OF_LINE)
        return (f"{BOS}{style.strip()}{EOM}{EOM}"
                f"{BOL}{END_OF_LINE}{lyrics_text}{EOM}"
                f"<|reserved_special_token_{duration_min}|>{EOT}{BOA}")
    return (f"{BOS}{style.strip()}{EOM}Instrumental{EOM}"
            f"<|reserved_special_token_{duration_min}|>{EOT}{BOA}")


def audio_ids_from_codes(codes) -> torch.Tensor:
    """codes: int array [frames, 2] = (layer-0, layer-1) codes in 0..1023.
    Returns the backbone's interleaved stream q0, q1, q0, q1, ... as vocab ids
    (same layout as modal_khala_layer_features.py, which was verified end to end)."""
    codes = torch.as_tensor(codes, dtype=torch.long)
    ids = torch.empty(codes.shape[0] * 2, dtype=torch.long)
    ids[0::2] = codes[:, 0] + VQ0_START_ID
    ids[1::2] = codes[:, 1] + VQ0_START_ID + Q1_OFFSET
    return ids


def build_sequence(prompt_ids, codes) -> tuple[torch.Tensor, int]:
    """Prompt followed by audio, cut to the backbone's 16,384-token window.
    Returns (ids, audio_start). Cutting keeps whole (q0, q1) frames."""
    audio = audio_ids_from_codes(codes)
    room = CONTEXT_LEN - len(prompt_ids)
    room -= room % 2
    ids = torch.cat([torch.as_tensor(prompt_ids, dtype=torch.long), audio[:room]])
    return ids, len(prompt_ids)


# ---- LoRA: small trainable add-ons next to frozen weights ---------------------------------
class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B(A x). W and b stay frozen; only A and B train.
    B starts at zero, so a fresh adapter changes nothing."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.enabled = True
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    # KhalaMLP adds gate/up bias a second time (swiglu_double_bias) through `.bias`,
    # so the wrapper must still expose the frozen layer's bias and weight.
    @property
    def bias(self):
        return self.base.bias

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            delta = (x.to(self.A.dtype) @ self.A.t()) @ self.B.t()
            out = out + (delta * self.scale).to(out.dtype)
        return out


def add_lora(model, r: int = 16, alpha: float = 32, targets=LORA_TARGETS) -> list[LoRALinear]:
    """Freeze every original weight, then wrap the chosen linears in all 24 layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for layer in model.layers:
        for parent in (layer.attn, layer.mlp):
            for name in targets:
                base = getattr(parent, name, None)
                if isinstance(base, nn.Linear):
                    lora = LoRALinear(base, r, alpha)
                    setattr(parent, name, lora)
                    wrapped.append(lora)
    return wrapped


@contextlib.contextmanager
def lora_disabled(wrapped: list[LoRALinear]):
    """Temporarily turn the add-ons off: the model is then exactly the original Shao."""
    for m in wrapped:
        m.enabled = False
    try:
        yield
    finally:
        for m in wrapped:
            m.enabled = True


def lora_state(wrapped: list[LoRALinear]) -> dict:
    return {f"{i}.A": m.A.detach().cpu() for i, m in enumerate(wrapped)} | \
           {f"{i}.B": m.B.detach().cpu() for i, m in enumerate(wrapped)}


@torch.no_grad()
def merge_lora(wrapped: list[LoRALinear]) -> None:
    """Bake the add-ons into the frozen weights (for exporting a normal checkpoint)."""
    for m in wrapped:
        m.base.weight += (m.scale * (m.B @ m.A)).to(m.base.weight.dtype)
        m.B.zero_()


# ---- scoring: how likely does the model find this song? ---------------------------------
def hidden_states(model, ids: torch.Tensor, use_checkpoint: bool) -> torch.Tensor:
    """Same computation as KhalaModel.forward_hidden_states (causal, positions 0..S-1),
    but each layer can be checkpointed: activations are recomputed during backward
    instead of stored, which is what lets 16k-token songs fit in GPU memory."""
    h = model.embed(ids)
    positions = torch.arange(ids.shape[1], device=ids.device)
    cos, sin = model._rope_at(positions, h.device, h.dtype)
    for layer in model.layers:
        if use_checkpoint:
            h = checkpoint(layer, h, cos, sin, use_reentrant=False)
        else:
            h = layer(h, cos, sin)
    return model.norm(h)


def _chunk_logp(h_chunk: torch.Tensor, head_w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = (h_chunk @ head_w.t()).float()                  # [n, REAL_VOCAB]
    return logits.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(logits, -1)


def audio_logp(model, ids: torch.Tensor, audio_start: int, use_checkpoint: bool = True,
               chunk: int = 1024) -> tuple[torch.Tensor, int]:
    """Sum over audio tokens of log p(token | everything before it), and how many there are.
    Prompt tokens are context only, never scored. The softmax runs over the real vocab
    (130,304 ids), the same set the generator samples from (khala_runtime.sample_backbone)."""
    h = hidden_states(model, ids[None], use_checkpoint)[0]  # [S, H]
    pred = h[audio_start - 1:-1]                             # position t predicts token t+1
    targets = ids[audio_start:]
    head_w = model.lm_head.weight[:REAL_VOCAB]
    total = torch.zeros((), device=h.device, dtype=torch.float32)
    for s in range(0, len(targets), chunk):
        args = (pred[s:s + chunk], head_w, targets[s:s + chunk])
        if use_checkpoint and torch.is_grad_enabled():
            lp = checkpoint(_chunk_logp, *args, use_reentrant=False)  # never store 130k-wide logits
        else:
            lp = _chunk_logp(*args)
        total = total + lp.sum()
    return total, len(targets)


# ---- the DPO objective ------------------------------------------------------------------
def dpo_loss(pol_w, pol_l, ref_w: float, ref_l: float, beta: float, nll_weight: float):
    """All four inputs are AVERAGE log-probability per audio token (log p per token):
    pol_* from the model being trained (carry gradients), ref_* from original Shao.

    reward(song) = (pol - ref) * TOKENS_PER_MINUTE  -> "how much more likely than original
                   Shao finds it, per minute of music" (so long songs don't get bigger votes)
    DPO loss     = -log sigmoid(beta * (reward(winner) - reward(loser)))
    NLL term     = -pol_w (plain "imitate the winner"; keeps both songs from sinking together)
    """
    chosen = (pol_w - ref_w) * TOKENS_PER_MINUTE
    rejected = (pol_l - ref_l) * TOKENS_PER_MINUTE
    margin = beta * (chosen - rejected)
    dpo = -F.logsigmoid(margin)
    nll = -pol_w
    loss = dpo + nll_weight * nll
    stats = {k: float(v.detach()) for k, v in {
        "loss": loss, "dpo": dpo, "nll_winner": nll,
        "reward_winner": chosen, "reward_loser": rejected, "margin": margin,
    }.items()}
    stats["correct"] = float(stats["margin"] > 0)
    return loss, stats
