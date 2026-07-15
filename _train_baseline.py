import time
import torch


from model import GPT, Config
import tokenizer as tokenizer_mod

torch.manual_seed(1337)
device = "cpu"

text = open(
    r"c:\Users\kesha\OneDrive\Desktop\test\llm_handout\llm_handout\data\train_corpus.txt",
    encoding="utf-8",
).read()
tok = tokenizer_mod.load()
ids = torch.tensor(tok.encode(text), dtype=torch.long)
print(
    f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens (vocab {tok.vocab_size})"
)

cfg = Config()
cfg.vocab_size = tok.vocab_size
model = GPT(cfg).to(device)
n = model.n_params()
print(f"model: {n:,} params")

opt = torch.optim.Adam(model.parameters(), lr=3e-4)


def get_batch(ids, block, batch):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i : i + block] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + block] for i in ix])
    return x, y


model.train()
t0 = time.time()
losses = []
for step in range(1, 2001):
    x, y = get_batch(ids, cfg.block_size, 8)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % 100 == 0 or step == 1:
        avg = sum(losses[-100:]) / len(losses[-100:])
        print(
            f"step {step:5d}  loss {avg:.4f}  ({(time.time()-t0)/step*1000:.0f} ms/step)"
        )

torch.save(
    {
        "model": model.state_dict(),
        "config": {
            k: getattr(cfg, k)
            for k in dir(cfg)
            if not k.startswith("_") and not callable(getattr(cfg, k))
        },
        "steps": 2000,
        "train_loss_curve": losses,
    },
    r"c:\Users\kesha\OneDrive\Desktop\test\project\baseline_ckpt.pt",
)
print(f"saved baseline_ckpt.pt ({time.time()-t0:.0f}s total)")
