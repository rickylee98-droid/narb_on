"""
NARB-ON Experiment — Results Plotting
=======================================
Generates the two required plots from §6 of the training spec:
  1. val_ppl vs tokens-seen — both arms (both seeds, shaded if available)
  2. λ/α trajectories — NARB arm only

Usage:
  # After Stage 0 completes both arms:
  python plot.py --narb_csv  out/stage0_narb/metrics.csv \
                 --softmax_csv out/stage0_softmax/metrics.csv \
                 --output  out/stage0_combined.png

  # Stage 1:
  python plot.py --narb_csv  out/stage1_narb_seed42/metrics.csv \
                              out/stage1_narb_seed7/metrics.csv \
                 --softmax_csv out/stage1_softmax_seed42/metrics.csv \
                               out/stage1_softmax_seed7/metrics.csv \
                 --output  out/stage1_combined.png
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


def load_csv(path: str) -> list:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--narb_csv',    nargs='+', default=[],
                   help="metrics.csv file(s) for NARB arm")
    p.add_argument('--softmax_csv', nargs='+', default=[],
                   help="metrics.csv file(s) for softmax arm")
    p.add_argument('--output',      default='combined_results.png')
    p.add_argument('--micro_batch', type=int, default=4)
    p.add_argument('--grad_accum',  type=int, default=4)
    p.add_argument('--seq_len',     type=int, default=1024)
    return p.parse_args()


def steps_to_tokens(steps, micro_batch, grad_accum, seq_len):
    return [int(s) * micro_batch * grad_accum * seq_len for s in steps]


def plot_results(args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("NARB-ON vs Softmax — Head-to-Head (Stage 0 / Stage 1)", fontsize=13)
    ax_ppl, ax_lam = axes

    colors = {'narb': '#2ecc71', 'softmax': '#3498db'}

    # ── Plot 1: val PPL vs tokens ─────────────────────────────────────────
    def plot_arm(csv_paths, label, color):
        all_ppls = defaultdict(list)
        for path in csv_paths:
            if not os.path.exists(path):
                print(f"  [WARN] not found: {path}")
                continue
            rows = load_csv(path)
            for row in rows:
                if 'val_ppl' in row and row['val_ppl']:
                    step = int(row['step'])
                    all_ppls[step].append(float(row['val_ppl']))

        if not all_ppls:
            return

        steps   = sorted(all_ppls.keys())
        tokens  = steps_to_tokens(steps, args.micro_batch, args.grad_accum, args.seq_len)
        means   = [np.mean(all_ppls[s]) for s in steps]
        mins_   = [np.min(all_ppls[s])  for s in steps]
        maxs_   = [np.max(all_ppls[s])  for s in steps]

        ax_ppl.plot(tokens, means, label=label, color=color, linewidth=2)
        if any(len(v) > 1 for v in all_ppls.values()):
            ax_ppl.fill_between(tokens, mins_, maxs_, color=color, alpha=0.15)

    plot_arm(args.narb_csv,    'NARB-ON (no RoPE, 2-adic prior)', colors['narb'])
    plot_arm(args.softmax_csv, 'Softmax (RoPE, SDPA)',             colors['softmax'])

    ax_ppl.set_xlabel("Tokens seen")
    ax_ppl.set_ylabel("Validation Perplexity")
    ax_ppl.set_title("Quality: val PPL vs tokens")
    ax_ppl.legend()
    ax_ppl.grid(alpha=0.3)

    # ── Plot 2: λ and α trajectories (NARB only) ─────────────────────────
    if args.narb_csv:
        path = args.narb_csv[0]
        if os.path.exists(path):
            rows  = load_csv(path)
            steps = [int(r['step']) for r in rows]
            toks  = steps_to_tokens(steps, args.micro_batch, args.grad_accum, args.seq_len)

            # λ per layer
            lam_cols = [k for k in (rows[0].keys() if rows else []) if k.startswith('lam_')]
            for col in lam_cols:
                vals = [float(r[col]) for r in rows if col in r and r[col]]
                ax_lam.plot(toks[:len(vals)], vals, label=col, linewidth=1.5, alpha=0.8)

            ax_lam.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='init=0.5')
            ax_lam.set_xlabel("Tokens seen")
            ax_lam.set_ylabel("λ value (prior blending)")
            ax_lam.set_title("NARB prior ablation: λ per layer")
            ax_lam.legend(fontsize=7, ncol=2)
            ax_lam.set_ylim(0, 1.05)
            ax_lam.grid(alpha=0.3)

    plt.tight_layout()
    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[plot] saved → {out}")


if __name__ == "__main__":
    args = parse_args()
    plot_results(args)
