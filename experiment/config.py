"""
NARB-ON Training Experiment — Configuration
============================================
Source of truth for all hyperparameters (§2 of training spec).
Both Stage 0 (plumbing shakedown) and Stage 1 (result run) live here.
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Architecture
    d_model:    int   = 256
    n_layer:    int   = 4
    n_head:     int   = 8      # d_head = d_model // n_head  MUST equal 32
    max_seq:    int   = 1024
    vocab_size: int   = 50304  # GPT-2 BPE padded to multiple of 128

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert self.d_model // self.n_head == 32, \
            f"d_head must be 32 (NARB requirement); got {self.d_model // self.n_head}"


@dataclass
class TrainConfig:
    # Training
    arch:            str   = "narb"       # "narb" | "softmax"
    seed:            int   = 42
    max_steps:       int   = 1000
    micro_batch:     int   = 8            # sequences per GPU step
    grad_accum:      int   = 8            # effective batch ≈ micro_batch * grad_accum * max_seq tokens
    dtype:           str   = "float32"    # "bfloat16" on GPU with bf16 support, "float32" for CPU

    # Optimizer (AdamW, spec §2)
    lr:              float = 6e-4
    beta1:           float = 0.9
    beta2:           float = 0.95
    weight_decay:    float = 0.1
    grad_clip:       float = 1.0

    # Schedule: linear warmup → cosine to 10% of peak
    warmup_steps:    int   = 100
    min_lr_ratio:    float = 0.1          # min LR = min_lr_ratio * lr

    # Data
    data_dir:        str   = "data"       # relative to experiment/; tokenized shards here
    num_workers:     int   = 0

    # Logging / checkpointing
    eval_every:      int   = 100
    eval_iters:      int   = 50           # batches used to estimate val loss
    save_every:      int   = 500
    out_dir:         str   = "out"        # relative to experiment/
    log_csv:         str   = "metrics.csv"

    # Efficiency benchmark (separate from training)
    bench_seq_lens:  list  = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192, 16384])


# ── Preset configurations ────────────────────────────────────────────────────

def stage0_narb() -> tuple:
    """Stage 0 debug run — NARB arm (15M params, ~500-1000 steps)."""
    m = ModelConfig(d_model=256, n_layer=4, n_head=8, max_seq=1024)
    t = TrainConfig(arch="narb", seed=42, max_steps=1000,
                    micro_batch=4, grad_accum=4,
                    lr=6e-4, warmup_steps=100,
                    eval_every=100, save_every=500,
                    out_dir="out/stage0_narb")
    return m, t


def stage0_softmax() -> tuple:
    """Stage 0 debug run — softmax arm."""
    m = ModelConfig(d_model=256, n_layer=4, n_head=8, max_seq=1024)
    t = TrainConfig(arch="softmax", seed=42, max_steps=1000,
                    micro_batch=4, grad_accum=4,
                    lr=6e-4, warmup_steps=100,
                    eval_every=100, save_every=500,
                    out_dir="out/stage0_softmax")
    return m, t


def stage1_narb(seed: int = 42) -> tuple:
    """Stage 1 result run — NARB arm (125M params)."""
    m = ModelConfig(d_model=768, n_layer=12, n_head=24, max_seq=1024)
    t = TrainConfig(arch="narb", seed=seed, max_steps=6000,
                    micro_batch=8, grad_accum=8,
                    lr=6e-4, warmup_steps=700,
                    dtype="bfloat16",
                    eval_every=250, save_every=500,
                    out_dir=f"out/stage1_narb_seed{seed}")
    return m, t


def stage1_softmax(seed: int = 42) -> tuple:
    """Stage 1 result run — softmax arm (125M params)."""
    m = ModelConfig(d_model=768, n_layer=12, n_head=24, max_seq=1024)
    t = TrainConfig(arch="softmax", seed=seed, max_steps=6000,
                    micro_batch=8, grad_accum=8,
                    lr=6e-4, warmup_steps=700,
                    dtype="bfloat16",
                    eval_every=250, save_every=500,
                    out_dir=f"out/stage1_softmax_seed{seed}")
    return m, t
