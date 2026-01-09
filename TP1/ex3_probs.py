import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- Chargement du modèle et du tokenizer ---
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# --- 1. Probabilités conditionnelles pour chaque token ---
phrase = "Artificial intelligence is fascinating."
inputs = tokenizer(phrase, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # shape (1, seq_len, vocab_size)

# Softmax pour obtenir les probabilités
probs = torch.softmax(logits, dim=-1)

input_ids = inputs["input_ids"][0]
print("Probabilités conditionnelles par token :")
for t in range(1, len(input_ids)):
    tok_id = input_ids[t].item()
    p = probs[0, t-1, tok_id].item()
    tok_txt = tokenizer.decode([tok_id])
    print(t, repr(tok_txt), f"{p:.3e}")

# --- 2. Log-probabilité totale et perplexité ---
log_probs = torch.log_softmax(logits, dim=-1)

total_logp = 0.0
n = 0
for t in range(1, len(input_ids)):
    tok_id = input_ids[t].item()
    lp = log_probs[0, t-1, tok_id].item()
    total_logp += lp
    n += 1

avg_neg_logp = - total_logp / n
ppl = math.exp(avg_neg_logp)

print("\nLog-proba totale:", total_logp)
print("Avg negative log-proba:", avg_neg_logp)
print("Perplexity:", ppl)

# --- 3. Comparaison avec plusieurs phrases ---
phrases = [
    "Artificial intelligence is fascinating.",
    "Artificial fascinating intelligence is.",
    "L'intelligence artificielle est fascinante."
]

for phrase in phrases:
    inputs = tokenizer(phrase, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

    input_ids = inputs["input_ids"][0]
    total_logp = 0.0
    n = 0
    for t in range(1, len(input_ids)):
        tok_id = input_ids[t].item()
        lp = log_probs[0, t-1, tok_id].item()
        total_logp += lp
        n += 1

    avg_neg_logp = - total_logp / n
    ppl = math.exp(avg_neg_logp)

    print(f"\nPhrase: {phrase}")
    print("total_logp:", total_logp)
    print("avg_neg_logp:", avg_neg_logp)
    print("perplexity:", ppl)

# --- 4. Top-10 tokens suivants pour un préfixe ---
prefix = "Artificial intelligence is"
inp = tokenizer(prefix, return_tensors="pt")

with torch.no_grad():
    out = model(**inp)
    logits2 = out.logits  # shape (1, seq_len, vocab_size)

# Dernier pas de temps pour le prochain token
last_logits = logits2[0, -1, :]
last_probs = torch.softmax(last_logits, dim=-1)

topk = 10
vals, idx = torch.topk(last_probs, k=topk)

print("\nTop-10 tokens probables après le préfixe :")
for p, tid in zip(vals.tolist(), idx.tolist()):
    print(repr(tokenizer.decode([tid])), f"{p:.3e}")