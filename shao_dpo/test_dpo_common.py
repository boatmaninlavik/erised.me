"""Local checks for dpo_common.py — runs on a laptop CPU in seconds, no GPU, no downloads.

Uses a tiny random KhalaModel (same code as Shao, just 2 small layers) to check that the
training math is wired correctly. Run: python3 shao_dpo/test_dpo_common.py
"""
import ast
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "khala_runtime"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dpo_common as dc                                   # noqa: E402
from core.khala_config import KhalaConfig                 # noqa: E402
from core.khala_model import KhalaModel                   # noqa: E402


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  (' + detail + ')') if detail else ''}")
    if not ok:
        sys.exit(1)


def shao_compose_prompt_text():
    """Pull compose_prompt_text + its constants straight out of backend_worker.py
    (importing the whole worker needs FastAPI/Megatron, so extract just these)."""
    src = (ROOT / "khala_runtime/backend/backend_worker.py").read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name == "compose_prompt_text")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in
                ("BOS", "EOM", "BOL", "EOT", "BOA", "END_OF_LINE") for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "bw", "exec"), ns)
    return ns["compose_prompt_text"]


# 1. prompt text identical to Shao's own app
shao = shao_compose_prompt_text()
lyr = "[Verse]\nline one\nline two\n\n[Chorus]\nhook"
for instrumental in (False, True):
    ours = dc.build_prompt_text("dream pop, female vocals", lyr, instrumental, 3)
    theirs = shao(lyrics=lyr, genre="", language="Instrumental" if instrumental else "",
                  duration=3, tags="dream pop, female vocals", description="")
    check(f"prompt text matches Shao app ({'instrumental' if instrumental else 'vocal'})", ours == theirs)

# 2. tokenizer: Shao's loader (AutoTokenizer) and every special token -> one id
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(ROOT / "khala_runtime/models/Tokenizer"), local_files_only=True)
ids = tok.encode(dc.build_prompt_text("pop", lyr, False, 3), add_special_tokens=False)
check("prompt starts with BOS and ends with BOA", ids[0] == 128000 and ids[-1] == 128010, f"{ids[:2]}...{ids[-3:]}")
for m in range(1, 11):
    t = tok.encode(f"<|reserved_special_token_{m}|>", add_special_tokens=False)
    if len(t) != 1:
        check(f"duration token {m} is a single id", False, str(t))
check("duration tokens 1-10 are single ids", True)

# 3. audio interleave + 16k cut
codes = torch.tensor([[5, 7], [1023, 0]])
check("interleave q0,q1,q0,q1", dc.audio_ids_from_codes(codes).tolist() == [128261, 129287, 129279, 129280])
seq, a0 = dc.build_sequence(list(range(1001)), torch.randint(0, 1024, (9000, 2)))
check("sequence cut to 16,384 with whole frames", len(seq) == dc.CONTEXT_LEN - 1 and (len(seq) - a0) % 2 == 0,
      f"len {len(seq)}, audio {len(seq) - a0}")

# 4. tiny Shao-shaped model (same classes, 2 small layers, double-bias quirk on)
torch.manual_seed(0)
cfg = KhalaConfig(hidden_size=64, num_layers=2, num_attention_heads=4, num_query_groups=2,
                  head_dim=16, ffn_hidden_size=96)
model = KhalaModel(cfg).float().eval()
for n, p in model.named_parameters():
    if n.endswith("bias"):
        torch.nn.init.normal_(p, std=0.1)                   # non-zero biases so the quirk matters
prompt = tok.encode(dc.build_prompt_text("pop", lyr, False, 3), add_special_tokens=False)
ids, a0 = dc.build_sequence(prompt, torch.randint(0, 1024, (300, 2)))

with torch.no_grad():
    ref_h = model.forward_hidden_states(ids[None])
    check("our forward == KhalaModel.forward_hidden_states",
          torch.allclose(dc.hidden_states(model, ids[None], False), ref_h, atol=1e-5))
    check("checkpointed forward == plain forward",
          torch.allclose(dc.hidden_states(model, ids[None], True), ref_h, atol=1e-5))
    s, n = dc.audio_logp(model, ids, a0, use_checkpoint=False, chunk=100)
    logits = model.lm_head(ref_h[0])[:, :dc.REAL_VOCAB].float()
    naive = F.log_softmax(logits, -1)[a0 - 1:-1].gather(-1, ids[a0:, None]).sum()
    check("chunked audio log-prob == naive full log-softmax", torch.allclose(s, naive, atol=1e-3),
          f"{float(s):.3f} vs {float(naive):.3f}, {n} tokens")
    check("only audio tokens scored", n == 600)

# 5. LoRA: fresh adapters change nothing; .bias still reaches the double-bias MLP
loras = dc.add_lora(model, r=4, alpha=8)
check("LoRA wrapped 7 linears x 2 layers", len(loras) == 14)
with torch.no_grad():
    check("fresh LoRA == original model",
          torch.allclose(dc.hidden_states(model, ids[None], False), ref_h, atol=1e-5))
trainable = [p for p in model.parameters() if p.requires_grad]
check("only LoRA A/B are trainable", len(trainable) == 28)

# 6. DPO at the start: policy == reference -> loss = log 2 exactly
w_ids, w0 = ids, a0
l_ids, l0 = dc.build_sequence(prompt, torch.randint(0, 1024, (250, 2)))
with torch.no_grad(), dc.lora_disabled(loras):
    rw = dc.audio_logp(model, w_ids, w0, False)
    rl = dc.audio_logp(model, l_ids, l0, False)
ref_w, ref_l = float(rw[0]) / rw[1], float(rl[0]) / rl[1]
pw = dc.audio_logp(model, w_ids, w0, True)
pl = dc.audio_logp(model, l_ids, l0, True)
loss, st = dc.dpo_loss(pw[0] / pw[1], pl[0] / pl[1], ref_w, ref_l, beta=0.1, nll_weight=0.0)
check("DPO loss at start == log 2", abs(float(loss.detach()) - math.log(2)) < 1e-4, f"{float(loss.detach()):.5f}")

# 7. a few steps: margin grows, frozen weights untouched, gradients only reach LoRA
frozen_before = model.layers[0].attn.q_proj.base.weight.clone()
opt = torch.optim.AdamW(trainable, lr=1e-2)
for _ in range(5):
    pw = dc.audio_logp(model, w_ids, w0, True)
    pl = dc.audio_logp(model, l_ids, l0, True)
    loss, st = dc.dpo_loss(pw[0] / pw[1], pl[0] / pl[1], ref_w, ref_l, beta=0.1, nll_weight=0.2)
    opt.zero_grad()
    loss.backward()
    opt.step()
check("training raises winner-vs-loser margin", st["margin"] > 0, f"margin {st['margin']:.3f}")
check("original weights unchanged", torch.equal(frozen_before, model.layers[0].attn.q_proj.base.weight))
with torch.no_grad(), dc.lora_disabled(loras):
    again = dc.audio_logp(model, w_ids, w0, False)
check("adapters off -> exactly original Shao again", abs(float(again[0]) / again[1] - ref_w) < 1e-6)

# 8. merging the adapters into the weights gives the same model
with torch.no_grad():
    before = dc.hidden_states(model, ids[None], False)
    dc.merge_lora(loras)
    after = dc.hidden_states(model, ids[None], False)
check("merged weights == weights + adapters", torch.allclose(before, after, atol=1e-4))
print("\nall checks passed")
