"""Optimized trainer.

Improvements over baseline:
  - BPE tokenizer pre-training (calls tokenizer.build_and_save if needed)
  - AdamW optimizer with weight decay (0.1 on 2-D params, 0 on bias/LN)
  - Gradient clipping (max_norm=1.0)
  - Cosine annealing LR with linear warmup (100 steps warmup, then cosine
    decay to 10% of peak LR by step 2000)
  - Larger batch size (16) to get better gradient estimates per step
  - Prints compressed-sequence stats after tokenization

HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_tokens.npy")


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i : i + block] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, warmup, max_steps, peak_lr, min_lr_ratio=0.1):
    """
    Linear warmup for `warmup` steps, then cosine decay to
    `peak_lr * min_lr_ratio` by `max_steps`.
    """
    if step < warmup:
        return peak_lr * step / warmup
    if step >= max_steps:
        return peak_lr * min_lr_ratio
    # cosine decay
    progress = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * coeff)


def make_optimizer(model, weight_decay=0.1, lr=3e-4):
    """
    AdamW with weight decay applied to 2-D tensors only.
    Biases and LayerNorm weight/bias are excluded.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument(
        "--skip_bpe_train",
        action="store_true",
        help="Skip BPE training if bpe_vocab.json already exists",
    )
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"

    torch.manual_seed(args.seed)
    device = "cpu"

    # ── Tokenizer ──────────────────────────────────────────────────────────
    vocab_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bpe_vocab.json"
    )
    if not os.path.exists(vocab_file) or not args.skip_bpe_train:
        tok = tokenizer_mod.build_and_save(args.data)
    else:
        tok = tokenizer_mod.load()
        print(f"Loaded BPE tokenizer (vocab_size={tok.vocab_size})")

    # ── Data (with caching) ────────────────────────────────────────────────
    text = open(args.data, encoding="utf-8").read()
    n_bytes = len(text.encode("utf-8"))

    if os.path.exists(CACHE_FILE):
        print(f"Loading cached tokens from {CACHE_FILE} ...")
        ids = torch.from_numpy(np.load(CACHE_FILE).astype(np.int64))
        print(
            f"corpus: {n_bytes:,} bytes -> {len(ids):,} cached tokens "
            f"(vocab {tok.vocab_size}, "
            f"compression {n_bytes/len(ids):.2f}x)"
        )
    else:
        print("Encoding corpus (one-time; result will be cached) ...")
        t_enc = time.time()
        ids_list = tok.encode(text)
        ids = torch.tensor(ids_list, dtype=torch.long)
        np.save(CACHE_FILE, np.array(ids_list, dtype=np.int32))
        print(
            f"corpus: {n_bytes:,} bytes -> {len(ids):,} tokens "
            f"(vocab {tok.vocab_size}, "
            f"compression {n_bytes/len(ids):.2f}x, "
            f"encode took {time.time()-t_enc:.1f}s, cached)"
        )

    # ── Model ──────────────────────────────────────────────────────────────
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} unique params (tie_weights={cfg.tie_weights})")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} unique params; got {n:,}"

    # ── Optimizer & LR Schedule ────────────────────────────────────────────
    opt = make_optimizer(model, weight_decay=0.1, lr=args.lr)

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    losses = []

    for step in range(1, args.steps + 1):
        # Update LR
        lr = get_lr(step, args.warmup, args.steps, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_steps = 8
        loss_accum = 0.0
        for micro_step in range(accum_steps):
            x, y = get_batch(ids, cfg.block_size, args.batch, device)
            _, loss = model(x, y)
            loss = loss / accum_steps
            loss.backward()
            loss_accum += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        losses.append(loss_accum)
        if step % args.log_every == 0 or step == 1:
            window = losses[-args.log_every :]
            avg = sum(window) / len(window)
            print(
                f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  "
                f"({(time.time()-t0)/step*1000:.0f} ms/step)"
            )

    # ── Save checkpoint ────────────────────────────────────────────────────
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                k: getattr(cfg, k)
                for k in dir(cfg)
                if not k.startswith("_") and not callable(getattr(cfg, k))
            },
            "steps": args.steps,
            "train_loss_curve": losses,
        },
        args.out,
    )
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
