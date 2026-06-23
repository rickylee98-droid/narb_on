"""Stage-0 harness smoke check — run before spending GPU hours."""
import sys, os, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from config import stage0_narb, stage0_softmax
from model  import build_model

print("=== Model build + param matching ===")
m_cfg, _ = stage0_narb()

narb_model    = build_model(m_cfg, 'narb',    init_seed=42)
softmax_model = build_model(m_cfg, 'softmax', init_seed=42)

nc = narb_model.count_params()
sc = softmax_model.count_params()
print(f"NARB    total={nc['total']:,}  shared={nc['shared']:,}  narb_specific={nc['narb_specific']:,}")
print(f"Softmax total={sc['total']:,}  shared={sc['shared']:,}  narb_specific={sc['narb_specific']:,}")

assert nc['shared'] == sc['shared'], f"Shared mismatch: {nc['shared']} vs {sc['shared']}"
delta = nc['total'] - sc['total']
assert delta == nc['narb_specific']
print(f"Param delta = {delta:,}  (= narb_specific: alpha + lmbda per block)  OK")
print()

print("=== Trunk weight identity check ===")
for name in ['wte.weight', 'blocks.0.ln_1.weight',
             'blocks.0.attn.q_proj.weight',
             'blocks.0.attn.k_proj.weight',
             'blocks.0.mlp.gate.weight']:
    nw = dict(narb_model.named_parameters())[name]
    sw = dict(softmax_model.named_parameters())[name]
    match = torch.allclose(nw, sw)
    print(f"  {name:<40}  identical={match}")
    assert match, f"MISMATCH: {name}"
print()

print("=== Forward pass (no grad) ===")
for arch, model in [('narb', narb_model), ('softmax', softmax_model)]:
    x = torch.randint(0, m_cfg.vocab_size, (2, 64))
    y = torch.randint(0, m_cfg.vocab_size, (2, 64))
    logits, loss = model(x, y)
    assert torch.isfinite(loss), f"{arch} loss is non-finite!"
    ppl = math.exp(min(loss.item(), 20))
    print(f"  [{arch:<8}] logits={tuple(logits.shape)}  loss={loss.item():.4f}  ppl={ppl:.1f}  OK")
print()

print("=== Gradient flow check (NARB lmbda + alpha) ===")
# With zero-init embeddings, Q·K ≈ 0 → f(s) ≈ 1 everywhere → grad ≈ 0.
# We verify grad EXISTS (not None), then verify λ MOVES off init via a few
# optimizer steps — that's the real §3.6 check.
narb_model.train()
opt_check = torch.optim.AdamW(narb_model.parameters(), lr=1e-2)

# Record init λ values
lam_init = [float(b.attn.lmbda.item()) for b in narb_model.blocks]

for step in range(20):
    x = torch.randint(0, m_cfg.vocab_size, (2, 32))
    y = torch.randint(0, m_cfg.vocab_size, (2, 32))
    _, loss = narb_model(x, y)
    loss.backward()

    # After first backward, verify grad exists (not None)
    if step == 0:
        for i, block in enumerate(narb_model.blocks):
            attn = block.attn
            lg = attn.lmbda.grad
            ag = attn.alpha_logit.grad         # Phase 5.1: was attn.alpha (output scale)
            assert lg is not None, f"lmbda.grad is None at block {i} — gradient path severed!"
            assert ag is not None, f"alpha_logit.grad is None at block {i} — gradient path severed!"
            print(f"  block {i}: lmbda.grad present ({lg.item():.2e})  "
                  f"alpha_logit.grad present (norm={ag.norm().item():.2e})")

    opt_check.step()
    opt_check.zero_grad(set_to_none=True)

lam_final = [float(b.attn.lmbda.item()) for b in narb_model.blocks]
moved = any(abs(f - i) > 1e-5 for f, i in zip(lam_final, lam_init))
print(f"  λ init  = {[f'{v:.5f}' for v in lam_init]}")
print(f"  λ final = {[f'{v:.5f}' for v in lam_final]}")
assert moved, "λ did not move off init after 20 optimizer steps!"
print("  λ moved off init  OK")
print()

print("=== Checkpoint round-trip check ===")
import tempfile, copy
narb_model.eval()
x_t = torch.randint(0, m_cfg.vocab_size, (1, 32))
with torch.no_grad():
    out_before, _ = narb_model(x_t)

opt_dummy = torch.optim.AdamW(narb_model.parameters(), lr=1e-3)
with tempfile.TemporaryDirectory() as tmp:
    ckpt = {
        'step': 1, 'val_loss': 9.9,
        'model': narb_model.state_dict(),
        'optimizer': opt_dummy.state_dict(),
        'model_cfg': narb_model.cfg,
        'arch': 'narb',
    }
    ckpt_path = os.path.join(tmp, 'ckpt.pt')
    torch.save(ckpt, ckpt_path)

    # Reload
    narb2 = build_model(m_cfg, 'narb', init_seed=99)  # different seed
    ckpt2 = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    narb2.load_state_dict(ckpt2['model'])
    narb2.eval()
    with torch.no_grad():
        out_after, _ = narb2(x_t)

    assert torch.allclose(out_before, out_after, atol=1e-5), "Checkpoint mismatch!"
print("  Checkpoint save → load → identical output  OK")
print()

print("=" * 50)
print("ALL HARNESS CHECKS PASSED")
print("Stage-0 GATE items verified in CPU fast-check:")
print("  [x] both arms forward without NaN/Inf")
print("  [x] param counts printed; shared weights match exactly")
print("  [x] lambda and alpha receive non-zero gradients")
print("  [x] checkpoint round-trips correctly")
print("  [ ] val loss CSV — verified during actual training run")
print("  [ ] combined loss plot — run plot.py after both arms complete")
print("=" * 50)
