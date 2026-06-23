"""
NARB-ON Experiment — Data Pipeline
====================================
Source: FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT subset).
Tokenizer: GPT-2 BPE via tiktoken. Output: uint16 .bin shards.

Usage:
  # Prepare data (run once):
  python data.py --stage 0        # ~100M tokens, Stage-0 slice
  python data.py --stage 1        # ~3B tokens, Stage-1 full run

  # Or use synthetic data for quick harness tests (no download needed):
  python data.py --synthetic

Shard format: flat uint16 little-endian, 1 file per shard.
First shard = val, rest = train (nanoGPT convention).
"""

import argparse
import os
import struct
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 50304       # GPT-2 BPE padded to multiple of 128
SHARD_TOKENS = 100_000_000   # 100M tokens per shard
STAGE0_SHARDS = 1            # ~100M tokens total for Stage 0
STAGE1_SHARDS = 30           # ~3B tokens total for Stage 1


def write_uint16_shard(tokens: np.ndarray, path: str) -> None:
    """Write token array to a uint16 little-endian binary file."""
    assert tokens.dtype == np.uint16
    with open(path, 'wb') as f:
        f.write(tokens.tobytes())
    print(f"  Wrote {len(tokens):,} tokens → {path}")


def read_uint16_shard(path: str) -> np.ndarray:
    """Read a uint16 binary shard."""
    return np.frombuffer(open(path, 'rb').read(), dtype=np.uint16)


# ── Synthetic data (no download, for harness testing) ─────────────────────────

def make_synthetic(data_dir: str, n_tokens: int = 10_000_000,
                   seed: int = 42) -> None:
    """
    Generate synthetic random-token data.  Useful to test the training harness
    without downloading FineWeb-Edu.  Do NOT use for real experiments.
    """
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    val_n = max(n_tokens // 20, 100_000)
    val_tokens = rng.integers(0, VOCAB_SIZE, size=val_n, dtype=np.uint16)
    write_uint16_shard(val_tokens, os.path.join(data_dir, "val.bin"))
    # train uses train_0000.bin to match ShardedTokenDataset glob pattern
    train_tokens = rng.integers(0, VOCAB_SIZE, size=n_tokens, dtype=np.uint16)
    write_uint16_shard(train_tokens, os.path.join(data_dir, "train_0000.bin"))

    print(f"[data] Synthetic data written to {data_dir}/")
    print("  NOTE: synthetic data is meaningless for loss comparison.")
    print("  Run `python data.py --stage 0` to get real FineWeb-Edu data.")


# ── Real data — FineWeb-Edu via Hugging Face ───────────────────────────────────

def tokenize_fineweb(data_dir: str, n_shards: int = STAGE0_SHARDS,
                     seed: int = 42) -> None:
    """
    Download and tokenize FineWeb-Edu sample-10BT.
    Requires: pip install datasets tiktoken huggingface_hub
    """
    try:
        import tiktoken
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Missing dependencies. Run:\n"
            "  pip install tiktoken datasets huggingface_hub"
        )

    os.makedirs(data_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    print(f"[data] Downloading FineWeb-Edu (sample-10BT), {n_shards} shard(s)…")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # Shuffle deterministically, then tokenize
    shard_idx = 0
    buf       = []
    buf_count = 0
    val_saved = False

    for doc in ds:
        text   = doc.get("text", "")
        tokens = enc.encode_ordinary(text)
        if len(tokens) == 0:
            continue
        tokens.append(enc.eot_token)          # GPT-2 EOT = 50256
        buf.extend(tokens)
        buf_count += len(tokens)

        while buf_count >= SHARD_TOKENS:
            chunk     = np.array(buf[:SHARD_TOKENS], dtype=np.uint16)
            buf       = buf[SHARD_TOKENS:]
            buf_count = len(buf)

            if not val_saved:
                write_uint16_shard(chunk, os.path.join(data_dir, "val.bin"))
                val_saved = True
            else:
                write_uint16_shard(chunk, os.path.join(data_dir, f"train_{shard_idx:04d}.bin"))
                shard_idx += 1

            if shard_idx >= n_shards:
                print(f"[data] Done. {shard_idx} train shard(s) + 1 val shard.")
                return

    # Flush remainder
    if buf and not val_saved:
        chunk = np.array(buf, dtype=np.uint16)
        write_uint16_shard(chunk, os.path.join(data_dir, "val.bin"))
    elif buf and shard_idx < n_shards:
        chunk = np.array(buf, dtype=np.uint16)
        write_uint16_shard(chunk, os.path.join(data_dir, f"train_{shard_idx:04d}.bin"))

    print(f"[data] Done. {shard_idx} train shard(s) + 1 val shard.")


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset, IterableDataset


class ShardedTokenDataset(IterableDataset):
    """
    Infinite iterable dataset over uint16 token shards.
    Deterministic data order: seeded once at construction, both arms share
    identical batch sequences in identical order (spec §1 control).
    """
    def __init__(self, data_dir: str, split: str,
                 seq_len: int, seed: int = 42):
        super().__init__()
        self.seq_len  = seq_len
        self.seed     = seed

        if split == 'val':
            path = os.path.join(data_dir, 'val.bin')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Val shard not found: {path}\n"
                    "Run: python data.py --stage 0  (or --synthetic)"
                )
            self.shards = [path]
        else:
            import glob
            self.shards = sorted(glob.glob(os.path.join(data_dir, 'train_*.bin')))
            if not self.shards:
                raise FileNotFoundError(
                    f"No train shards in {data_dir}/\n"
                    "Run: python data.py --stage 0  (or --synthetic)"
                )

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        shard_order = list(range(len(self.shards)))

        while True:
            rng.shuffle(shard_order)
            for si in shard_order:
                tokens = read_uint16_shard(self.shards[si]).astype(np.int64)
                n      = len(tokens)

                # shuffle start positions within shard
                starts = np.arange(0, n - self.seq_len - 1, self.seq_len)
                rng.shuffle(starts)

                for start in starts:
                    x = torch.from_numpy(tokens[start   : start + self.seq_len    ])
                    y = torch.from_numpy(tokens[start+1 : start + self.seq_len + 1])
                    yield x, y


class ValDataset(Dataset):
    """Fixed-order val dataset for reproducible perplexity estimates."""
    def __init__(self, data_dir: str, seq_len: int, max_batches: int = 200):
        path   = os.path.join(data_dir, 'val.bin')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Val shard not found: {path}\n"
                "Run: python data.py --stage 0"
            )
        tokens = read_uint16_shard(path).astype(np.int64)
        n      = min(len(tokens), max_batches * seq_len + 1)
        tokens = tokens[:n]
        self.xs = []
        self.ys = []
        for i in range(0, n - seq_len - 1, seq_len):
            self.xs.append(torch.tensor(tokens[i   : i + seq_len]))
            self.ys.append(torch.tensor(tokens[i+1 : i + seq_len + 1]))
            if len(self.xs) >= max_batches:
                break

    def __len__(self):  return len(self.xs)
    def __getitem__(self, i): return self.xs[i], self.ys[i]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare data for NARB-ON training experiment")
    p.add_argument('--stage',     type=int, default=0, choices=[0, 1],
                   help="Stage 0 (~100M tok) or Stage 1 (~3B tok)")
    p.add_argument('--synthetic', action='store_true',
                   help="Generate synthetic random data (harness testing only)")
    p.add_argument('--data_dir',  type=str, default='data',
                   help="Output directory for tokenized shards")
    args = p.parse_args()

    if args.synthetic:
        make_synthetic(args.data_dir)
    else:
        n_shards = STAGE0_SHARDS if args.stage == 0 else STAGE1_SHARDS
        tokenize_fineweb(args.data_dir, n_shards=n_shards)
