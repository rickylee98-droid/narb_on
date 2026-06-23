"""
NARB-ON: Minimal character-level LM wrapper  (§5 of build spec).

Architecture:
  token_embeddings  : Embedding(vocab, d_model)
  n_layers × block  : h = h + (1/√n_layers) · NARBLinearAttention(h)
  final_norm         : RMSNorm(d_model)
  lm_head            : Linear(d_model, vocab, bias=False)  — weight-tied

No DB, no server, no external tokenizer.  torch-only.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from narb_linear_attention import NARBLinearAttention, D_HEAD_REQUIRED


# ── RMSNorm ──────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms * self.weight).to(x.dtype)


# ── Trivial char-level tokenizer (~10 lines) ──────────────────────────────────
VOCAB = (" abcdefghijklmnopqrstuvwxyz"
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "0123456789.,!?\n")
_PAD_ID = 0
_UNK_ID = 1
_C2I = {c: i + 2 for i, c in enumerate(VOCAB)}
_I2C = {v: k for k, v in _C2I.items()}
VOCAB_SIZE = len(_C2I) + 2     # +2 for PAD, UNK

def encode(text: str):
    return [_C2I.get(c, _UNK_ID) for c in text]

def decode(ids):
    return "".join(_I2C.get(i, "") for i in ids)


# ── NARBModel ─────────────────────────────────────────────────────────────────
class NARBModel(nn.Module):
    """
    Minimal NARB-ON language model for smoke testing.
    d_model must be a multiple of D_HEAD_REQUIRED (=32).
    n_heads = d_model // 32.
    """

    def __init__(self,
                 d_model:   int = 64,
                 n_layers:  int = 2,
                 vocab_size: int = VOCAB_SIZE):
        super().__init__()
        if d_model % D_HEAD_REQUIRED != 0:
            raise ValueError(
                f"d_model ({d_model}) must be a multiple of {D_HEAD_REQUIRED} (d_head)."
            )
        n_heads = d_model // D_HEAD_REQUIRED
        self.d_model  = d_model
        self.n_layers = n_layers

        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # Residual scale: 1/√n_layers (spec §5)
        self.res_scale = 1.0 / math.sqrt(n_layers)

        self.blocks = nn.ModuleList([
            NARBLinearAttention(d_model, n_heads)
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)

        # lm_head: tied to embedding weight (spec §5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids : (B, N) int64
        returns   : (B, N, vocab_size) logits
        """
        h = self.token_embeddings(token_ids)        # (B, N, d_model)
        for block in self.blocks:
            h = h + self.res_scale * block(h)       # residual: spec §5
        h = self.final_norm(h)
        return self.lm_head(h)


# ── Smoke test (test 9) ───────────────────────────────────────────────────────
def train_smoke(n_steps: int = 60, verbose: bool = True) -> list:
    """
    Overfit one short string for n_steps with AdamW + cross-entropy.
    Verifies:
      - loss decreases
      - λ stays in [0, 1] at every step
      - no NaN/Inf at any step
    Returns list of per-step losses.
    """
    model = NARBModel(d_model=64, n_layers=2)
    model.train()

    text = "the quick brown fox jumps over the lazy dog"
    ids  = torch.tensor([encode(text)], dtype=torch.long)   # (1, N)
    x, y = ids[:, :-1], ids[:, 1:]                          # (1, N-1)

    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    losses = []

    for step in range(n_steps):
        logits = model(x)                                    # (1, N-1, vocab)
        loss   = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        assert torch.isfinite(loss), f"NaN/Inf loss at step {step}"

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Verify λ stays in [0, 1] for every attention block
        for blk in model.blocks:
            lam = float(blk.lmbda)
            assert 0.0 <= lam <= 1.0 + 1e-6, \
                f"λ={lam:.4f} out of [0,1] at step {step}"

        losses.append(loss.item())
        if verbose and (step + 1) % 10 == 0:
            lam0 = model.blocks[0].lmbda.clamp(0, 1).item()
            grad0 = model.blocks[0].lmbda.grad
            grad_str = f"{grad0.item():.2e}" if grad0 is not None else "None"
            print(f"  step {step+1:3d}  loss={loss.item():.4f}  "
                  f"λ₀={lam0:.5f}  ∂λ₀={grad_str}")

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )
    if verbose:
        print(f"\n  Smoke test PASSED: {losses[0]:.4f} → {losses[-1]:.4f}")

    return losses


if __name__ == "__main__":
    print("NARB-ON  Test 9: end-to-end smoke")
    print("=" * 50)
    train_smoke(n_steps=60, verbose=True)
