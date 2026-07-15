# Run Log

| Run # | Hypothesis | Vocab Size | Params | Train Loss (final) | Dev BPB |
|-------|-----------|------------|--------|-------------------|---------|
| 0     | Baseline: byte tokenizer, Adam constant LR, 4-layer, n_embd=160 | 256 | 1,339,840 | 1.7315 | **2.3718** |
| 1     | BPE 512 + weight tying + AdamW + cosine LR + grad clip + block_size=256 | 512 | 1,929,280 | 2.3689 | **1.7528** |

## Notes
- Baseline: 2000 steps, 85-125 ms/step, total ~179s
- BPE tokenizer 512: 256 merges, 2.20× overall compression, 2.85× on Hindi
- Corpus: 7,318,592 bytes → 3,323,573 tokens (BPE) vs 7,318,592 tokens (byte)
