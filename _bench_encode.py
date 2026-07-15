import time, sys
sys.path.insert(0, '.')
import tokenizer as T

tok = T.load()
text = open(r'c:\Users\kesha\OneDrive\Desktop\test\llm_handout\llm_handout\data\train_corpus.txt', encoding='utf-8').read()
nb = len(text.encode('utf-8'))
print(f'Corpus size: {len(text):,} chars, {nb:,} bytes')
t0 = time.time()
ids = tok.encode(text)
dt = time.time() - t0
ratio = nb / len(ids)
print(f'Encode time: {dt:.2f}s -> {len(ids):,} tokens ({ratio:.2f}x compression)')
