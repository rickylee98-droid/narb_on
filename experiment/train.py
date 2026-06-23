"""
NARB-ON Training Experiment — Main Training Script
=====================================================
Fork of nanoGPT's training loop with the attention block as the only swap.
Both arms (narb, softmax) use this identical script.

Run Stage 0:
  python train.py --arch narb     --stage 0
  python train.py --arch softmax  --stage 0

Run Stage 1 (on GPU, after Stage 0 is fully green):
  python train.py --arch narb    --stage 1 --seed 42
  python train.py --arch softmax --stage 1 --seed 42

Stage 0 GATE checklist (§5):
  [x] loss descends smoothly, both arms, no NaN/Inf
  [x] printed param counts match (within α,λ scalars)
  [x] val loss computes; checkpoint round-trips
  [x] λ and α move off init → see metrics.csv
  [x] combined loss plot → run plot.py after both Stage-0 jobs complete
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from config   import ModelConfig, TrainConfig, stage0_narb, stage0_softmax, \
                     stage1_narb, stage1_softmax
from model    import build_model, NARBTransformer
from data     import ShardedTokenDataset, ValDataset


# ── LR schedule (linear warmup → cosine to min_lr) ───────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine LR schedule with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    if step >= cfg.max_steps:
        return cfg.lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + coeff * (1.0 - cfg.min_lr_ratio))


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: NARBTransformer, optimizer: torch.optim.Optimizer,
                    step: int, val_loss: float, out_dir: str) -> None:
    ckpt = {
        'step':       step,
        'val_loss':   val_loss,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'model_cfg':  model.cfg,
        'arch':       model.arch,
    }
    path = os.path.join(out_dir, f"ckpt_{step:06d}.pt")
    torch.save(ckpt, path)
    # Also save as 'latest' for easy resume
    torch.save(ckpt, os.path.join(out_dir, "ckpt_latest.pt"))
    print(f"  [ckpt] saved step={step} val_loss={val_loss:.4f} → {path}")


def load_checkpoint(path: str, model: NARBTransformer,
                    optimizer: torch.optim.Optimizer) -> int:
    ckpt      = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    step      = ckpt['step']
    val_loss  = ckpt['val_loss']
    print(f"  [ckpt] resumed from step={step} val_loss={val_loss:.4f}")
    return step


# ── Metric collection helpers ─────────────────────────────────────────────────

def collect_narb_params(model: NARBTransformer) -> dict:
    """Collect λ per layer and α per head (Phase 5.1) from NARB arm for CSV."""
    lam_vals   = {}
    alpha_vals = {}
    for i, block in enumerate(model.blocks):
        attn = block.attn
        if hasattr(attn, 'lmbda'):
            lam_vals[f"lam_L{i}"] = float(attn.lmbda.clamp(0,1).item())
        if hasattr(attn, 'alpha_logit'):           # Phase 5.1 temperature
            alphas = torch.sigmoid(attn.alpha_logit).detach()
            for h, a in enumerate(alphas):
                alpha_vals[f"alpha_L{i}_H{h}"] = float(a.item())
    return {**lam_vals, **alpha_vals}


@torch.no_grad()
def estimate_val_loss(model: NARBTransformer, val_loader: DataLoader,
                      device: torch.device, dtype, n_iters: int = 50) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= n_iters: break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


# ── Main training loop ────────────────────────────────────────────────────────

def train(m_cfg: ModelConfig, t_cfg: TrainConfig,
          resume: str = None) -> None:
    # ── Device / dtype ────────────────────────────────────────────────────
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                 'float32':  torch.float32}
    pt_dtype  = dtype_map.get(t_cfg.dtype, torch.float32)
    if pt_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[train] bf16 not supported on this device; falling back to fp32.")
        pt_dtype = torch.float32

    print(f"[train] device={device}  dtype={pt_dtype}  arch={t_cfg.arch}")

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(t_cfg.seed)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(m_cfg, t_cfg.arch, init_seed=42).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Separate decay / no-decay param groups (nanoGPT convention)
    decay_params    = [p for n, p in model.named_parameters()
                       if p.dim() >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.dim() < 2  and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params,    'weight_decay': t_cfg.weight_decay},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=t_cfg.lr, betas=(t_cfg.beta1, t_cfg.beta2),
        fused=(device.type == 'cuda'),
    )

    # ── Checkpoint resume ─────────────────────────────────────────────────
    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, model, optimizer)

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), t_cfg.data_dir)
    if not os.path.exists(data_dir):
        print(f"[train] data_dir not found: {data_dir}")
        print("  Run: python data.py --synthetic  (quick) or  --stage 0 (real)")
        sys.exit(1)

    train_ds   = ShardedTokenDataset(data_dir, 'train', m_cfg.max_seq, t_cfg.seed)
    val_ds     = ValDataset(data_dir, m_cfg.max_seq, max_batches=t_cfg.eval_iters * 4)
    train_iter = iter(DataLoader(train_ds, batch_size=t_cfg.micro_batch,
                                  num_workers=t_cfg.num_workers, pin_memory=(device.type=='cuda')))
    val_loader = DataLoader(val_ds, batch_size=t_cfg.micro_batch, shuffle=False)

    # ── Output dir + CSV ──────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), t_cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, t_cfg.log_csv)
    csv_fields_init = False

    print(f"[train] out_dir={out_dir}")
    print(f"[train] effective batch ≈ {t_cfg.micro_batch * t_cfg.grad_accum * m_cfg.max_seq:,} tokens/step")
    print(f"[train] total steps={t_cfg.max_steps}  eval_every={t_cfg.eval_every}")
    print(f"[train] training START (step {start_step} → {t_cfg.max_steps})")
    print()

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    t0 = time.perf_counter()

    for step in range(start_step, t_cfg.max_steps):
        # LR update
        lr = get_lr(step, t_cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── Gradient accumulation ─────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro in range(t_cfg.grad_accum):
            x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type,
                                dtype=pt_dtype,
                                enabled=(pt_dtype != torch.float32)):
                _, loss = model(x, y)

            loss     = loss / t_cfg.grad_accum
            loss_accum += loss.item()
            loss.backward()

        # ── Gradient clip ─────────────────────────────────────────────────
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
        if not math.isfinite(float(grad_norm)):
            print(f"  [WARN] non-finite grad norm at step {step} — skipping update")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()

        # ── Step timing ───────────────────────────────────────────────────
        t1 = time.perf_counter()
        tok_per_s = (t_cfg.micro_batch * t_cfg.grad_accum * m_cfg.max_seq) / (t1 - t0 + 1e-9)
        t0 = t1

        # ── Console log ───────────────────────────────────────────────────
        if step % max(1, t_cfg.eval_every // 5) == 0:
            print(f"  step {step:5d}/{t_cfg.max_steps}  "
                  f"loss={loss_accum:.4f}  lr={lr:.2e}  "
                  f"|g|={grad_norm:.3f}  {tok_per_s:,.0f} tok/s")

        # ── Eval + CSV + checkpoint ───────────────────────────────────────
        if step % t_cfg.eval_every == 0 or step == t_cfg.max_steps - 1:
            val_loss = estimate_val_loss(model, val_loader, device, pt_dtype,
                                        n_iters=t_cfg.eval_iters)
            val_ppl  = math.exp(min(val_loss, 20))

            print(f"\n  ── eval step {step} ──")
            print(f"     train_loss={loss_accum:.4f}  "
                  f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")

            # Collect NARB-specific parameters for CSV
            row = {
                'step':       step,
                'train_loss': round(loss_accum, 6),
                'val_loss':   round(val_loss, 6),
                'val_ppl':    round(val_ppl, 4),
                'lr':         round(lr, 8),
                'grad_norm':  round(float(grad_norm), 6),
                'tok_per_s':  round(tok_per_s, 1),
                'arch':       t_cfg.arch,
            }
            if t_cfg.arch == 'narb':
                narb_row = collect_narb_params(model)
                row.update(narb_row)
                # Print a summary of λ and α ranges
                lam_vals   = [v for k, v in narb_row.items() if k.startswith('lam_')]
                alpha_vals = [v for k, v in narb_row.items() if k.startswith('alpha_')]
                if lam_vals:
                    print(f"     λ range: [{min(lam_vals):.4f}, {max(lam_vals):.4f}]")
                if alpha_vals:
                    print(f"     α range: [{min(alpha_vals):.4f}, {max(alpha_vals):.4f}]")

            print()

            # CSV append
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not csv_fields_init:
                    writer.writeheader()
                    csv_fields_init = True
                writer.writerow(row)

        # ── Checkpoint ────────────────────────────────────────────────────
        if step > 0 and (step % t_cfg.save_every == 0 or step == t_cfg.max_steps - 1):
            val_loss = estimate_val_loss(model, val_loader, device, pt_dtype,
                                        n_iters=t_cfg.eval_iters)
            save_checkpoint(model, optimizer, step, val_loss, out_dir)

        # ── NaN guard ─────────────────────────────────────────────────────
        if not math.isfinite(loss_accum):
            print(f"[FATAL] non-finite loss at step {step}: {loss_accum}. Aborting.")
            sys.exit(1)

    print(f"\n[train] DONE. Metrics → {csv_path}")
    print(f"[train] Run `python plot.py --out_dir {out_dir}` to plot results.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NARB-ON training experiment")
    p.add_argument('--arch',    required=True, choices=['narb', 'softmax'],
                   help="Which attention arm to train")
    p.add_argument('--stage',   type=int, default=0, choices=[0, 1],
                   help="Stage 0 (debug, 15M, ~1k steps) or Stage 1 (125M)")
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--resume',  type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument('--data_dir',    type=str, default=None,
                   help="Override data directory")
    p.add_argument('--max_steps',   type=int, default=None,
                   help="Override max training steps (useful for CPU gate-check demo)")
    p.add_argument('--micro_batch', type=int, default=None,
                   help="Override micro-batch size")
    p.add_argument('--grad_accum',  type=int, default=None,
                   help="Override gradient accumulation steps")
    p.add_argument('--eval_every',  type=int, default=None,
                   help="Override eval interval")
    p.add_argument('--max_seq',     type=int, default=None,
                   help="Override max sequence length (cuts seq_len for fast CPU demo)")
    args = p.parse_args()

    if args.stage == 0:
        m_cfg, t_cfg = (stage0_narb() if args.arch == 'narb' else stage0_softmax())
    else:
        m_cfg, t_cfg = (stage1_narb(args.seed) if args.arch == 'narb'
                        else stage1_softmax(args.seed))

    t_cfg.seed = args.seed
    if args.data_dir:    t_cfg.data_dir    = args.data_dir
    if args.max_steps:   t_cfg.max_steps   = args.max_steps
    if args.micro_batch: t_cfg.micro_batch = args.micro_batch
    if args.grad_accum:  t_cfg.grad_accum  = args.grad_accum
    if args.eval_every:  t_cfg.eval_every  = args.eval_every
    if args.max_seq:     m_cfg = ModelConfig(
        d_model=m_cfg.d_model, n_layer=m_cfg.n_layer,
        n_head=m_cfg.n_head, max_seq=args.max_seq,
        vocab_size=m_cfg.vocab_size)

    train(m_cfg, t_cfg, resume=args.resume)
