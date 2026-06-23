"""
NARB-ON Experiment Model
========================
Shared trunk for both arms.  One variable: the attention block.
Everything else is byte-identical via name-keyed deterministic init (§3).

Arm A: NARBLinearAttentionPhase5   (built kernel + per-head α)
Arm B: SoftmaxAttention            (SDPA + RoPE, no learned pos emb)

Spec §3 controls:
  - No wpe (no learned positional embeddings in either arm)
  - SwiGLU MLP, hidden = ⌈8/3·d_model⌉ rounded to 64, same both arms
  - Pre-norm RMSNorm, same both arms
  - wte weight tied to lm_head, same both arms
  - All trunk weights (wte, MLP, norms, q/k/v/out projs) initialised by
    name-keyed seed so both arms start from byte-identical trunk weights
  - Only α (per head), λ (per layer) differ — NARB-only, <0.01% of params
"""

import math
import sys
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from narb_linear_attention import NARBLinearAttention, D_HEAD_REQUIRED
# Phase 5.1 α is now in NARBLinearAttention itself (alpha_logit per head).
# NARBLinearAttentionPhase5 is retired — use NARBLinearAttention directly.
from config import ModelConfig


# ────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ────────────────────────────────────────────────────────────────────────────

def _swiglu_hidden(d_model: int, multiple_of: int = 64) -> int:
    """Round 8/3·d to the next multiple_of."""
    h = int(math.ceil(8 / 3 * d_model))
    return ((h + multiple_of - 1) // multiple_of) * multiple_of


def _stable_hash(s: str) -> int:
    """Stable string hash (not Python's built-in, which varies across runs)."""
    h = 0
    for c in s.encode('utf-8'):
        h = (h * 31 + c) & 0xFFFFFFFF
    return h


# ────────────────────────────────────────────────────────────────────────────
# Building blocks (shared by both arms)
# ────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms * self.weight).to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU MLP: hidden = ⌈8/3·d_model⌉, rounded to 64. Same for both arms."""
    def __init__(self, d_model: int):
        super().__init__()
        hidden     = _swiglu_hidden(d_model)
        self.gate  = nn.Linear(d_model, hidden, bias=False)
        self.up    = nn.Linear(d_model, hidden, bias=False)
        self.down  = nn.Linear(hidden,  d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ────────────────────────────────────────────────────────────────────────────
# Arm B: Softmax attention with RoPE (the strong baseline)
# ────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Rotary position encoding (RoPE) — buffers only, no learned params."""
    def __init__(self, d_head: int, max_seq: int = 4096, base: float = 10_000.0):
        super().__init__()
        theta = base ** (-2.0 * torch.arange(0, d_head, 2).float() / d_head)
        t     = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, theta)                # (max_seq, d/2)
        # Full-dim cos/sin by repeating both halves
        cos   = torch.cat([freqs.cos(), freqs.cos()], dim=-1)   # (max_seq, d)
        sin_  = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer('cos_table', cos)
        self.register_buffer('sin_table', sin_)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        N   = q.shape[2]
        cos = self.cos_table[:N].unsqueeze(0).unsqueeze(0)   # (1,1,N,d)
        sin = self.sin_table[:N].unsqueeze(0).unsqueeze(0)
        q   = q * cos + self._rotate_half(q) * sin
        k   = k * cos + self._rotate_half(k) * sin
        return q, k


class SoftmaxAttention(nn.Module):
    """
    Arm B: standard causal MHA with RoPE and SDPA/FlashAttention.
    No learned positional embeddings (spec §3 — no wpe in either arm).
    """
    def __init__(self, d_model: int, n_heads: int, max_seq: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head  = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        # Shared param names — must match NARBLinearAttentionPhase5 exactly
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.d_head, max_seq)

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        H, d    = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)   # (B,H,N,d)
        K = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        Q, K = self.rope(Q, K)

        # SDPA: uses FlashAttention when available on CUDA
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)


# ────────────────────────────────────────────────────────────────────────────
# Arm A: NARB attention + Phase-5 per-head α
# ────────────────────────────────────────────────────────────────────────────

# NARBLinearAttentionPhase5 is retired.
# Phase 5.1 per-head temperature α is now built into NARBLinearAttention.alpha_logit.
# The experiment uses NARBLinearAttention directly.


# ────────────────────────────────────────────────────────────────────────────
# Transformer block (shared structure, attention module is the only swap)
# ────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    Pre-norm transformer block: h = h + attn(norm(h)) + mlp(norm(h)).
    'attn' is the ONLY difference between arms.
    """
    def __init__(self, d_model: int, attn: nn.Module):
        super().__init__()
        self.ln_1 = RMSNorm(d_model)
        self.attn = attn
        self.ln_2 = RMSNorm(d_model)
        self.mlp  = SwiGLU(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # Phase 5.2: pre-norm only — norm applied to input BEFORE each sub-layer;
        # NO extra norm is added inside NARBLinearAttention (Q,K are L2-normed
        # inside _prep_qkv; that is the kernel's own device, not a layer norm).
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ────────────────────────────────────────────────────────────────────────────
# Full model
# ────────────────────────────────────────────────────────────────────────────

class NARBTransformer(nn.Module):
    """
    Shared transformer trunk.  attn_factory decides NARB vs softmax.
    No wpe — positional info comes from RoPE (softmax) or 2-adic prior (NARB).
    """
    def __init__(self, cfg: ModelConfig, arch: str):
        super().__init__()
        self.cfg  = cfg
        self.arch = arch

        def make_attn():
            if arch == 'narb':
                # Phase 5.1 α is built into NARBLinearAttention.alpha_logit
                return NARBLinearAttention(cfg.d_model, cfg.n_head)
            elif arch == 'softmax':
                return SoftmaxAttention(cfg.d_model, cfg.n_head, cfg.max_seq)
            else:
                raise ValueError(f"Unknown arch: {arch!r}. Use 'narb' or 'softmax'.")

        # Trunk — identical structure for both arms
        self.wte    = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, make_attn()) for _ in range(cfg.n_layer)
        ])
        self.ln_f   = RMSNorm(cfg.d_model)

        # lm_head weight-tied to wte (no bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx: Tensor,
                targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        idx     : (B, N) int64
        targets : (B, N) int64 or None
        Returns (logits (B,N,V), loss or None)
        """
        x      = self.wte(idx)                    # (B, N, d_model)
        for block in self.blocks:
            x  = block(x)
        x      = self.ln_f(x)
        logits = self.lm_head(x)                  # (B, N, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def count_params(self) -> Dict[str, int]:
        """Return dict: total, shared (excludes α/λ), narb_specific."""
        narb_keys = {'lmbda', 'alpha_logit'}
        total, shared, narb_specific = 0, 0, 0
        for name, p in self.named_parameters():
            n = p.numel()
            total += n
            key = name.split('.')[-2] if '.' in name else name.split('.')[-1]
            if any(k in name for k in narb_keys):
                narb_specific += n
            else:
                shared += n
        return {'total': total, 'shared': shared, 'narb_specific': narb_specific}


# ────────────────────────────────────────────────────────────────────────────
# Name-keyed deterministic init  (§3 "identical trunk weights")
# ────────────────────────────────────────────────────────────────────────────

_NARB_SPECIFIC = frozenset({
    'lmbda', 'alpha_logit',              # Phase 5.1: alpha_logit replaces old 'alpha'
    '_triu_rows', '_triu_cols', '_triu_w',
    '_ones_masked_C', '_T_masked_C',
})
_SOFTMAX_SPECIFIC = frozenset({'cos_table', 'sin_table'})
_SKIP_PATTERNS   = _NARB_SPECIFIC | _SOFTMAX_SPECIFIC


def name_keyed_init(model: NARBTransformer,
                    base_seed: int = 42) -> None:
    """
    Initialise all *shared* parameters deterministically by name so that
    both arms start from byte-identical trunk weights (spec §3).

    Each parameter's seed = base_seed + stable_hash(param_name).
    Arm-specific params (lmbda, alpha) are skipped → keep their default init.
    RMSNorm weights initalised to 1 (standard). Residual output projections
    scaled by 1/√(2·n_layer) (nanoGPT convention).
    """
    n_layer = model.cfg.n_layer

    for name, param in model.named_parameters():
        # Skip buffers (not in named_parameters) and arm-specific params
        if any(skip in name for skip in _SKIP_PATTERNS):
            continue
        # Skip lm_head — weight-tied to wte, initialised below with wte
        if name == 'lm_head.weight':
            continue

        seed = (base_seed + _stable_hash(name)) & 0x7FFF_FFFF
        gen  = torch.Generator().manual_seed(seed)

        with torch.no_grad():
            if 'norm' in name.lower() and 'weight' in name:
                nn.init.ones_(param)                          # RMSNorm → 1
            elif param.dim() < 2:
                nn.init.zeros_(param)                         # bias / 1-d scalars
            elif 'out_proj' in name or ('down' in name and 'mlp' in name):
                # Residual branch projections: scale by 1/√(2·L)
                std = 0.02 / math.sqrt(2.0 * n_layer)
                nn.init.normal_(param, 0.0, std, generator=gen)
            else:
                nn.init.normal_(param, 0.0, 0.02, generator=gen)


# ────────────────────────────────────────────────────────────────────────────
# Convenience builder
# ────────────────────────────────────────────────────────────────────────────

def build_model(cfg: ModelConfig, arch: str,
                init_seed: int = 42) -> NARBTransformer:
    """Create, init, and return a model for the given arm."""
    model = NARBTransformer(cfg, arch)
    name_keyed_init(model, base_seed=init_seed)

    counts = model.count_params()
    print(f"[{arch.upper()}] params: total={counts['total']:,}  "
          f"shared={counts['shared']:,}  "
          f"narb_specific={counts['narb_specific']:,}")
    return model
