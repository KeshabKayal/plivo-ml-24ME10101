"""Pre-encode the training corpus and save to bpe_tokens.npy cache."""
import time, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tokenizer as T

DATA = r'c:\Users\kesha\OneDrive\Desktop\test\llm_handout\llm_handout\data\train_corpus.txt'
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bpe_tokens.npy')

tok = T.load()
print(f'Tokenizer loaded: vocab_size={tok.vocab_size}')
text = open(DATA, encoding='utf-8').read()
nb = len(text.encode('utf-8'))
print(f'Corpus: {nb:,} bytes, {len(text):,} chars')

t0 = time.time()
print('Encoding ...', flush=True)
ids = tok.encode(text)
dt = time.time() - t0
ratio = nb / len(ids)
print(f'Done in {dt:.1f}s: {nb:,} bytes -> {len(ids):,} tokens ({ratio:.2f}x compression)')

np.save(CACHE, np.array(ids, dtype=np.int32))
print(f'Cache saved: {CACHE}')

# Verify round-trip losslessness
dec = tok.decode(ids[:1000])
orig = text.encode('utf-8')[:len(dec.encode('utf-8'))].decode('utf-8', errors='replace')
# Just do a partial spot-check
sample_ids = ids[:10000]
sample_dec = tok.decode(sample_ids)
sample_enc = tok.encode(sample_dec)
print('Spot-check losslessness ...', end=' ')
if sample_enc == sample_ids:
    print('OK')
else:
    print('WARNING: spot check failed')
