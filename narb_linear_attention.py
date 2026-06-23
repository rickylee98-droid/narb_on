"""
NARB-ON: Exact O(N) Causal Linear Attention
============================================
Clean-room build per NARB ON LINEAR ATTENTION BUILD SPEC v2.

Locked config: d_head=32, C=64, L*=6, D_phi=561,
               degree-2 exact kernel, row-norm causal, fp32 states.
§3.6 mandatory: λ is a live differentiable tensor throughout forward/decode.

Build order:
  Phase 0  _dense_oracle()         -- O(N²) reference, TEST-ONLY
  Phase 1  kernel_coeffs / feature_map  -- exact polynomial feature map
  Phase 2  _sequential_forward()   -- O(N) sequential recurrence
  Phase 3  _chunked_forward()      -- O(N) chunked production path
  Phase 4  NARBLinearAttention.step() -- O(1)-per-token decode

Public API:  NARBLinearAttention,  kernel_coeffs,  feature_map
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ────────────────────────────────────────────────────────────────────────────
# Locked constants (§1)
# ────────────────────────────────────────────────────────────────────────────
D_HEAD_REQUIRED: int = 32
CHUNK_DEFAULT:   int = 64      # C = 2^6
L_STAR_DEFAULT:  int = 6
EPS_DEFAULT:   float = 1e-6


def _d_phi(d: int) -> int:
    """D_φ = 1 + d + d(d+1)/2. For d=32 → 561."""
    return 1 + d + d * (d + 1) // 2


# ────────────────────────────────────────────────────────────────────────────
# Phase 1a — kernel coefficients (§3.1 and §3.4)
# ────────────────────────────────────────────────────────────────────────────
def kernel_coeffs(d: int,
                  sigma_sq: float = 1.0,
                  include_gaussian: bool = False
                  ) -> Tuple[float, float, float]:
    """
    Returns (c0, c1, c2) so that f(s) = c0 + c1·s + c2·s².

    Default (exact):  f(s) = (s/√d + 1)² ≥ 0 for all s.
    Optional Gaussian: f(s) = (s/√d+1)²·exp(s/σ²), fitted by NNLS on [-1,1].
    Falls back to exact if NNLS yields any negative f value.
    """
    if not include_gaussian:
        return 1.0, 2.0 / math.sqrt(d), 1.0 / d

    try:
        import numpy as np
        from scipy.optimize import nnls as scipy_nnls
        grid   = np.linspace(-1.0, 1.0, 512)
        target = ((grid / math.sqrt(d) + 1.0) ** 2) * np.exp(grid / sigma_sq)
        basis  = np.stack([np.ones_like(grid), grid, grid ** 2], axis=1)
        coeffs, _ = scipy_nnls(basis, target)
        c0, c1, c2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        if (c0 + c1 * grid + c2 * grid ** 2).min() < 0.0:
            raise ValueError("NNLS fit has negative values; falling back to exact.")
        return c0, c1, c2
    except Exception as exc:
        warnings.warn(f"[NARB] Gaussian fit failed: {exc}. Using exact kernel.", stacklevel=2)
        return kernel_coeffs(d, include_gaussian=False)


# ────────────────────────────────────────────────────────────────────────────
# Phase 1b — exact feature map φ(x) ∈ ℝ^{D_φ}  (§3.2)
# ────────────────────────────────────────────────────────────────────────────
def _triu_info(d: int, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """Upper-triangle indices and weight vector for vech_w (√2 off-diag, 1 diag)."""
    rows, cols = torch.triu_indices(d, d, offset=0, device=device)
    w = torch.where(rows < cols,
                    torch.full(rows.shape, math.sqrt(2.0), device=device),
                    torch.ones(rows.shape, device=device))
    return rows, cols, w


def feature_map(x: Tensor,
                c0: float, c1: float, c2: float,
                triu_rows: Tensor,
                triu_cols: Tensor,
                triu_w:    Tensor,
                alpha: Optional[Tensor] = None) -> Tensor:
    """
    φ(x): (..., d) → (..., D_φ=561) in fp32.
    x must be L2-normalised over the last dim.
    Identity: φ(q)·φ(k) = c0 + c1·(q·k) + c2·(q·k)² = f(q·k)  (exact).

    Phase 5.1 — per-head temperature (alpha is not None):
      alpha: (H,) tensor, α_h = sigmoid(alpha_logit_h) ∈ (0,1).
      Kernel becomes f_h(s) = (α_h·s + 1)²  →  c1_h = 2α_h, c2_h = α_h².
      p1_h = √(2α_h)·x,  p2_h = α_h·vech_w(xxᵀ).
      x must have H at dim 1: shape (B,H,N,d) or (B,H,d).
      At α_h = 1/√d this reproduces v2 exactly (backward-compatible init).
    """
    x = x.float()

    if alpha is None:
        # ── v2 scalar path (backward-compatible) ─────────────────────────
        p0 = torch.full((*x.shape[:-1], 1), math.sqrt(c0),
                        dtype=torch.float32, device=x.device)
        p1 = math.sqrt(c1) * x
        outer = x[..., :, None] * x[..., None, :]
        p2 = math.sqrt(c2) * (outer[..., triu_rows, triu_cols] * triu_w)
    else:
        # ── Phase 5.1: per-head temperature ──────────────────────────────
        # alpha: (H,).  Reshape for broadcasting with x at dim 1.
        ndim = x.dim()
        if ndim == 4:       # (B, H, N, d)
            a = alpha.float().view(1, alpha.shape[0], 1, 1)
        elif ndim == 3:     # (B, H, d) — single-token decode
            a = alpha.float().view(1, alpha.shape[0], 1)
        else:
            raise ValueError(f"Per-head alpha needs x.dim() in (3,4); got {ndim}")

        # p0 = √c0·1 = 1  (c0 = 1, α-independent)
        p0 = torch.ones(*x.shape[:-1], 1, dtype=torch.float32, device=x.device)
        # p1 = √(2α)·x                     — gradient flows through a
        p1 = torch.sqrt(2.0 * a) * x
        # p2 = α·vech_w(xxᵀ)               — gradient flows through a
        outer = x[..., :, None] * x[..., None, :]
        p2 = a * (outer[..., triu_rows, triu_cols] * triu_w)

    return torch.cat([p0, p1, p2], dim=-1)


# ────────────────────────────────────────────────────────────────────────────
# 2-adic prior helpers
# ────────────────────────────────────────────────────────────────────────────
def _v2_matrix(n: int, device: torch.device) -> Tensor:
    """
    T[i,j] = 2^{-v2(|i-j|)} for i≠j,  T[i,i] = 1.
    Bit trick: lsb(m) = m & (-m) = 2^{v2(m)}, so 2^{-v2(m)} = 1/lsb(m).
    """
    idx  = torch.arange(n, dtype=torch.int64, device=device)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    safe = diff.clone(); safe[safe == 0] = 1
    lsb  = safe & (-safe)
    return 1.0 / lsb.float()


# ────────────────────────────────────────────────────────────────────────────
# Phase 0 — Dense causal oracle  (O(N²), TEST-ONLY)
# ────────────────────────────────────────────────────────────────────────────
def _dense_oracle(Q: Tensor, K: Tensor, V: Tensor,
                  lam: float, c0: float, c1: float, c2: float,
                  eps: float = EPS_DEFAULT) -> Tensor:
    """O(N²) exact reference. TEST-ONLY — never call from production."""
    B, H, N, _ = Q.shape
    Q, K, V = Q.float(), K.float(), V.float()
    S     = torch.matmul(Q, K.transpose(-2, -1))
    f_mat = c0 + c1 * S + c2 * S * S
    T     = _v2_matrix(N, Q.device)
    blend = (1.0 - lam) + lam * T
    cmask = torch.tril(torch.ones(N, N, device=Q.device))
    Omega = f_mat * blend * cmask
    num   = torch.matmul(Omega, V)
    den   = Omega.sum(-1, keepdim=True) + eps
    return num / den


# ────────────────────────────────────────────────────────────────────────────
# Phase 2 — Sequential O(N) recurrence
# ────────────────────────────────────────────────────────────────────────────
def _sequential_forward(Q: Tensor, K: Tensor, V: Tensor,
                        lam: float,
                        c0: float, c1: float, c2: float,
                        eps: float, l_star: int,
                        triu_rows: Tensor, triu_cols: Tensor,
                        triu_w: Tensor) -> Tensor:
    """O(N) sequential recurrence.  Matches Phase 0 when L* ≥ ceil(log₂ N)."""
    B, H, N, d = Q.shape
    d_v = V.shape[-1]
    D   = _d_phi(d)
    C   = 1 << l_star
    dev = Q.device
    Q, K, V = Q.float(), K.float(), V.float()
    out = torch.zeros(B, H, N, d_v, device=dev, dtype=torch.float32)

    for b in range(B):
      for h in range(H):
        q, k, v = Q[b, h], K[b, h], V[b, h]
        S_g = torch.zeros(D, d_v, device=dev)
        z_g = torch.zeros(D,      device=dev)
        S_L = [torch.zeros(1 << (l+1), D, d_v, device=dev) for l in range(l_star)]
        z_L = [torch.zeros(1 << (l+1), D,      device=dev) for l in range(l_star)]

        for i in range(N):
            phi_q = feature_map(q[i], c0, c1, c2, triu_rows, triu_cols, triu_w)
            phi_k = feature_map(k[i], c0, c1, c2, triu_rows, triu_cols, triu_w)
            a     = i % C

            # (1) Query past — λ-free dilated sum, then scale
            dil_n = 0.0; dil_d = 0.0
            for l in range(l_star):
                L     = l + 1
                c_cls = a % (1 << L)
                coeff = 0.5 ** L
                dil_n = dil_n + coeff * (phi_q @ S_L[l][c_cls])
                dil_d = dil_d + coeff * (phi_q @ z_L[l][c_cls])
            num = (phi_q @ S_g) - lam * dil_n
            den = (phi_q @ z_g) - lam * dil_d

            # (2) Self-term (prior T[i,i] = 1, λ-independent)
            f_ii = (phi_q * phi_k).sum()
            num  = num + f_ii * v[i]
            den  = den + f_ii

            out[b, h, i] = num / (den + eps)

            # (3) State update (λ-independent)
            outer_kv = phi_k.unsqueeze(-1) * v[i].unsqueeze(0)
            S_g = S_g + outer_kv; z_g = z_g + phi_k
            for l in range(l_star):
                L     = l + 1
                c_cls = a % (1 << L)
                S_L[l][c_cls] = S_L[l][c_cls] + outer_kv
                z_L[l][c_cls] = z_L[l][c_cls] + phi_k

    return out


# ────────────────────────────────────────────────────────────────────────────
# Phase 3 — Chunked parallel forward  (production path)
# ────────────────────────────────────────────────────────────────────────────
def _chunked_forward(Q:           Tensor,
                     K:           Tensor,
                     V:           Tensor,
                     lam:         Tensor,         # TENSOR — §3.6 mandatory
                     c0:          float,
                     c1:          float,
                     c2:          float,
                     eps:         float,
                     l_star:      int,
                     chunk:       int,
                     ones_masked: Tensor,          # tril(ones(C,C))  — M_0, λ-free buffer
                     T_masked:    Tensor,          # tril(T_local)    — 2-adic, λ-free buffer
                     triu_rows:   Tensor,
                     triu_cols:   Tensor,
                     triu_w:      Tensor,
                     alpha:       Optional[Tensor] = None) -> Tensor:  # Phase 5.1 per-head α
    """
    O(N) chunked production forward — FULLY DIFFERENTIABLE training path.
    §3.6: λ is a live tensor.  Cross-chunk carries are NOT detached so that
    gradients flow back through all past K/V to q/k/v/out_proj and alpha_logit.

    Carry design — list-of-lists of independent per-class tensors:
      carry_S_L[l][c] : (B,H,D,d_v)  accumulated out-of-place via `+`
      carry_z_L[l][c] : (B,H,D)
    Each class entry is a separate tensor → no shared storage → no version-
    counter aliasing → no need for .detach() or .clone() on the read side.
    The decode path (step()) keeps its own detached carries (gradients N/A).
    """
    B, H, N, d = Q.shape
    d_v = V.shape[-1]
    D   = _d_phi(d)
    C   = chunk
    dev = Q.device

    Q, K, V = Q.float(), K.float(), V.float()
    lam = lam.float().to(dev)   # tensor, grad flows

    out_chunks: list = []
    a_idx = torch.arange(C, dtype=torch.int64, device=dev)

    # Intra-chunk prior block (λ-differentiable, computed once).
    prior_C = (1.0 - lam) * ones_masked.to(dev) + lam * T_masked.to(dev)  # (C,C)

    # ── Carry state: per-class independent tensors (fully differentiable) ─
    # carry_S_L[l][c]: (B,H,D,d_v) — accumulated via out-of-place + (no detach).
    # carry_z_L[l][c]: (B,H,D)     — same.
    # Global channel uses standard tensors accumulated out-of-place.
    carry_S_g = torch.zeros(B, H, D, d_v, device=dev)
    carry_z_g = torch.zeros(B, H, D,      device=dev)
    carry_S_L = [[torch.zeros(B, H, D, d_v, device=dev) for _ in range(1 << (l+1))]
                 for l in range(l_star)]
    carry_z_L = [[torch.zeros(B, H, D,      device=dev) for _ in range(1 << (l+1))]
                 for l in range(l_star)]

    n_chunks = (N + C - 1) // C

    for chunk_idx in range(n_chunks):
        base = chunk_idx * C
        end  = min(base + C, N)
        T    = end - base

        Qc = Q[:, :, base:end]
        Kc = K[:, :, base:end]
        Vc = V[:, :, base:end]

        # ── (A) Intra-chunk ───────────────────────────────────────────────
        if T == C:
            prior = prior_C
        else:
            ones_T = torch.tril(torch.ones(T, T, device=dev))
            T_v2_T = torch.tril(_v2_matrix(T, dev))
            prior  = (1.0 - lam) * ones_T + lam * T_v2_T

        S_mat = torch.matmul(Qc, Kc.transpose(-2, -1))
        if alpha is None:
            f_mat = c0 + c1 * S_mat + c2 * S_mat * S_mat
        else:
            a_bc  = alpha.float().view(1, alpha.shape[0], 1, 1)
            f_mat = (a_bc * S_mat + 1.0) ** 2                  # (B,H,T,T)

        Omega_intra = f_mat * prior
        num_intra   = torch.matmul(Omega_intra, Vc)            # (B,H,T,d_v)
        den_intra   = Omega_intra.sum(-1)                       # (B,H,T)

        # ── (B) Cross-chunk: fully differentiable accumulation ────────────
        a_T  = a_idx[:T] + base
        PHIq = feature_map(Qc, c0, c1, c2, triu_rows, triu_cols, triu_w,
                           alpha=alpha)                         # (B,H,T,D)

        # Dilated sum λ-free (scalar class index keeps each lookup at
        # (B,H,D,d_v) = 2.3 MB not 184 MB).  index_add (out-of-place) scatters
        # per-class results into dil_num without in-place ops on grad tensors.
        dil_num = torch.zeros(B, H, T, d_v, device=dev)
        dil_den = torch.zeros(B, H, T,      device=dev)
        for l in range(l_star):
            L     = l + 1
            n_c   = 1 << L
            cls_l = (a_T % n_c).long()
            coeff = 0.5 ** L
            for c_val in range(n_c):
                mask  = (cls_l == c_val)
                if not mask.any(): continue
                idx_T  = mask.nonzero(as_tuple=False)[:, 0]   # (T_c,)
                PHIq_c = PHIq[:, :, mask, :]                   # (B,H,T_c,D)
                # Independent tensor — no .detach()/.clone() needed:
                S_c = carry_S_L[l][c_val]                      # (B,H,D,d_v)
                z_c = carry_z_L[l][c_val]                      # (B,H,D)
                dn_c = coeff * torch.einsum('bhtd,bhdv->bhtv', PHIq_c, S_c)
                dz_c = coeff * (PHIq_c * z_c.unsqueeze(2)).sum(-1)
                dil_num = dil_num + dil_num.new_zeros(B, H, T, d_v).index_add(
                    2, idx_T, dn_c)
                dil_den = dil_den + dil_den.new_zeros(B, H, T).index_add(
                    2, idx_T, dz_c)

        num_cross = torch.einsum('bhcd,bhdv->bhcv', PHIq, carry_S_g) - lam * dil_num
        den_cross = torch.einsum('bhcd,bhd->bhc',   PHIq, carry_z_g) - lam * dil_den

        num   = num_intra + num_cross
        den   = den_intra + den_cross
        out_c = num / (den.unsqueeze(-1) + eps)
        out_chunks.append(out_c)

        # ── (C) Carry update — out-of-place, fully differentiable ─────────
        PHIk = feature_map(Kc, c0, c1, c2, triu_rows, triu_cols, triu_w,
                           alpha=alpha)                         # (B,H,T,D)

        carry_S_g = carry_S_g + torch.einsum('bhcd,bhcv->bhdv', PHIk, Vc)
        carry_z_g = carry_z_g + PHIk.sum(dim=2)

        for l in range(l_star):
            L   = l + 1
            n_c = 1 << L
            cls = (a_T % n_c).long()
            for c_val in range(n_c):
                mask = (cls == c_val)
                if not mask.any(): continue
                PHIk_c = PHIk[:, :, mask, :]                   # (B,H,T_c,D)
                Vc_c   = Vc[:, :, mask, :]                     # (B,H,T_c,d_v)
                # Out-of-place + rebinds the list entry — no shared storage.
                carry_S_L[l][c_val] = (carry_S_L[l][c_val] +
                    torch.einsum('bhtd,bhtv->bhdv', PHIk_c, Vc_c))
                carry_z_L[l][c_val] = carry_z_L[l][c_val] + PHIk_c.sum(dim=2)

    return torch.cat(out_chunks, dim=2)   # (B, H, N, d_v)


# ────────────────────────────────────────────────────────────────────────────
# Public module: NARBLinearAttention  (§2)
# ────────────────────────────────────────────────────────────────────────────
class NARBLinearAttention(nn.Module):
    """
    O(N) causal linear attention block.  §3.6: λ is a live differentiable tensor.

    forward(x)        : (B,N,d_model) → (B,N,d_model)   — Phase 3 chunked
    step(x_t, state)  : (B,1,d_model) → (B,1,d_model)   — Phase 4 O(1)/token

    Requires d_head = d_model // n_heads == 32.
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 *,
                 chunk:            int   = CHUNK_DEFAULT,
                 l_star:           int   = L_STAR_DEFAULT,
                 include_gaussian: bool  = False,
                 sigma_sq:         float = 1.0,
                 eps:              float = EPS_DEFAULT):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        d_head = d_model // n_heads
        if d_head != D_HEAD_REQUIRED:
            raise ValueError(
                f"d_head must be {D_HEAD_REQUIRED} (spec §1). "
                f"Got d_model={d_model}, n_heads={n_heads} → d_head={d_head}."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_head
        self.chunk   = chunk
        self.l_star  = l_star
        self.eps     = eps

        c0, c1, c2 = kernel_coeffs(d_head, sigma_sq, include_gaussian)
        self.c0, self.c1, self.c2 = c0, c1, c2

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # λ — learnable blending parameter.  §3.6: NEVER call .item() here.
        self.lmbda = nn.Parameter(torch.tensor(0.5))

        # Phase 5.1 — per-head learnable temperature α = sigmoid(alpha_logit) ∈ (0,1).
        # Init: logit(1/√d_head) so α₀ = 1/√d → exactly reproduces v2 at startup.
        # Derivation: sigmoid(x)=1/√d ⟹ x = -log(√d - 1).
        init_logit = float(-math.log(math.sqrt(d_head) - 1.0))
        self.alpha_logit = nn.Parameter(torch.full((n_heads,), init_logit))

        # Feature map buffers
        rows, cols, w = _triu_info(d_head, device=torch.device('cpu'))
        self.register_buffer('_triu_rows', rows)
        self.register_buffer('_triu_cols', cols)
        self.register_buffer('_triu_w',    w)

        # Precomputed λ-independent C×C buffers (both lower-triangular / causal-masked).
        # Spec §3.6: bake ONLY the λ-free structure; combine with tensor λ at runtime.
        #   _ones_masked_C : tril(ones(C,C))  — M_0 channel
        #   _T_masked_C    : tril(T_local)    — 2-adic prior channel
        C        = chunk
        a_idx_C  = torch.arange(C, dtype=torch.int64)
        diff_C   = (a_idx_C.unsqueeze(0) - a_idx_C.unsqueeze(1)).abs()
        safe_C   = diff_C.clone(); safe_C[safe_C == 0] = 1
        T_loc_C  = 1.0 / (safe_C & (-safe_C)).float()           # (C,C), diag=1
        causal_C = torch.tril(torch.ones(C, C))
        self.register_buffer('_ones_masked_C', causal_C.clone())
        self.register_buffer('_T_masked_C',    T_loc_C * causal_C)

    # ── Internal helpers ──────────────────────────────────────────────────
    def _prep_qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Project, reshape (B,H,N,d_head), L2-normalise Q and K.
        Cast to fp32 so fp16 I/O works with fp32 weights (§7)."""
        B, N, _ = x.shape
        H, d    = self.n_heads, self.d_head
        x32     = x.float()

        def proj(lin: nn.Linear) -> Tensor:
            return lin(x32).view(B, N, H, d).transpose(1, 2)

        Q = F.normalize(proj(self.q_proj), p=2, dim=-1)
        K = F.normalize(proj(self.k_proj), p=2, dim=-1)
        V = proj(self.v_proj)
        return Q, K, V

    def _lam(self) -> Tensor:
        """Returns λ as a TENSOR (fp32). §3.6: never .item() / float()."""
        return self.lmbda.clamp(0.0, 1.0)

    def _alpha(self) -> Tensor:
        """
        Phase 5.1: per-head temperature α = sigmoid(alpha_logit) ∈ (0,1).
        At init: α = 1/√d_head → reproduces v2 feature map exactly.
        Gradient flows through sigmoid → alpha_logit is trainable.
        """
        return torch.sigmoid(self.alpha_logit)

    # ── Phase 3 forward ───────────────────────────────────────────────────
    def forward(self, x: Tensor) -> Tensor:
        """(B, N, d_model) → (B, N, d_model).  Phase 3 chunked O(N) path.
        Phase 5.2: no extra norm here — caller's pre-norm (RMSNorm on block input)
        is the only layer-level normalisation; Q,K are L2-normed inside _prep_qkv."""
        B, N, d_in = x.shape
        assert d_in == self.d_model, f"Input last dim {d_in} ≠ d_model {self.d_model}"
        if not torch.isfinite(x).all():
            raise ValueError("Input to NARBLinearAttention contains non-finite values.")

        Q, K, V = self._prep_qkv(x)
        lam   = self._lam()                                      # tensor, grad flows (§3.6)
        alpha = self._alpha()                                     # (H,) tensor, Phase 5.1

        out = _chunked_forward(
            Q, K, V, lam,
            self.c0, self.c1, self.c2,
            self.eps, self.l_star, self.chunk,
            self._ones_masked_C,
            self._T_masked_C,
            self._triu_rows, self._triu_cols, self._triu_w,
            alpha=alpha,
        )
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out).to(x.dtype)

    # ── Phase 4 decode step ───────────────────────────────────────────────
    def init_state(self, batch_size: int, device: torch.device) -> Dict:
        """Zero O(1) decode state (constant size regardless of sequence length)."""
        B, H, D, d_v = batch_size, self.n_heads, _d_phi(self.d_head), self.d_head
        return {
            'S_g':      torch.zeros(B, H, D, d_v, device=device),
            'z_g':      torch.zeros(B, H, D,       device=device),
            'S_L':      [torch.zeros(B, H, 1 << (l+1), D, d_v, device=device)
                         for l in range(self.l_star)],
            'z_L':      [torch.zeros(B, H, 1 << (l+1), D,       device=device)
                         for l in range(self.l_star)],
            'step_idx': 0,
        }

    def step(self, x_t: Tensor, state: Dict) -> Tuple[Tensor, Dict]:
        """
        O(1)-per-token decode.  §3.6: λ is a live tensor here too.
        x_t : (B, 1, d_model)  →  returns (out_t (B,1,d_model), new_state).
        """
        assert x_t.shape[1] == 1, "step() expects a single token per call"
        B   = x_t.shape[0]
        H   = self.n_heads
        d   = self.d_head
        dev = x_t.device

        Q, K, V = self._prep_qkv(x_t)
        q_t = Q[:, :, 0, :].float()
        k_t = K[:, :, 0, :].float()
        v_t = V[:, :, 0, :].float()

        lam   = self._lam().float().to(dev)                      # tensor, grad flows (§3.6)
        alpha = self._alpha().float().to(dev)                    # (H,) tensor, Phase 5.1
        D     = _d_phi(d)
        C     = 1 << self.l_star
        i     = state['step_idx']
        a     = i % C

        # q_t, k_t: (B, H, d) — feature_map handles 3D with per-head alpha
        phi_q = feature_map(q_t, self.c0, self.c1, self.c2,
                            self._triu_rows, self._triu_cols, self._triu_w,
                            alpha=alpha)                          # (B,H,D)
        phi_k = feature_map(k_t, self.c0, self.c1, self.c2,
                            self._triu_rows, self._triu_cols, self._triu_w,
                            alpha=alpha)

        S_g = state['S_g']; z_g = state['z_g']
        S_L = state['S_L']; z_L = state['z_L']

        # ── (1) Past contribution: dilated sum λ-free, scale once ─────────
        dil_num = torch.zeros(B, H, d, device=dev)
        dil_den = torch.zeros(B, H,    device=dev)
        for l in range(self.l_star):
            L     = l + 1
            c_cls = a % (1 << L)
            coeff = 0.5 ** L
            S_sel = S_L[l][:, :, c_cls, :, :]                   # (B,H,D,d_v)
            z_sel = z_L[l][:, :, c_cls, :]                      # (B,H,D)
            dil_num = dil_num + coeff * torch.einsum('bhd,bhdv->bhv', phi_q, S_sel)
            dil_den = dil_den + coeff * (phi_q * z_sel).sum(-1)

        num = torch.einsum('bhd,bhdv->bhv', phi_q, S_g) - lam * dil_num
        den = (phi_q * z_g).sum(-1)                         - lam * dil_den

        # ── (2) Self-term (prior T[i,i]=1, λ-independent) ─────────────────
        f_ii = (phi_q * phi_k).sum(-1, keepdim=True)            # (B,H,1)
        num  = num + f_ii * v_t
        den  = den + f_ii.squeeze(-1)

        out_t = num / (den.unsqueeze(-1) + self.eps)

        # ── (3) State update (λ-independent) ──────────────────────────────
        outer_kv = torch.einsum('bhd,bhv->bhdv', phi_k, v_t)
        new_S_g  = S_g + outer_kv
        new_z_g  = z_g + phi_k

        new_S_L, new_z_L = [], []
        for l in range(self.l_star):
            L     = l + 1
            c_cls = a % (1 << L)
            sL    = S_L[l].clone()
            zL    = z_L[l].clone()
            sL[:, :, c_cls, :, :] = sL[:, :, c_cls, :, :] + outer_kv
            zL[:, :, c_cls, :]    = zL[:, :, c_cls, :]    + phi_k
            new_S_L.append(sL)
            new_z_L.append(zL)

        new_state = {
            'S_g': new_S_g, 'z_g': new_z_g,
            'S_L': new_S_L, 'z_L': new_z_L,
            'step_idx': i + 1,
        }

        out_t = out_t.to(x_t.dtype)
        out_t = out_t.view(B, H * d).unsqueeze(1)
        return self.out_proj(out_t), new_state
