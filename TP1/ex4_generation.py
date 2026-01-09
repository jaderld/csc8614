import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SEED = 42 
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs_greedy = model.generate(
    **inputs,
    max_length=50
)
text_greedy = tokenizer.decode(outputs_greedy[0], skip_special_tokens=True)
print("Greedy decoding:")
print(text_greedy)
print("="*60)

def generate_once(seed, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=None):
    torch.manual_seed(seed)
    kwargs = {
        "max_length": 50,
        "do_sample": True,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p
    }
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("Sampling generations (5 seeds, temp=0.7, top-k=50, top-p=0.95):")
for s in [1, 2, 3, 4, 5]:
    print("SEED", s)
    print(generate_once(s))
    print("-"*40)

print("Sampling avec repetition_penalty=2.0:")
print("Seed 1, temp=0.7")
print(generate_once(seed=1, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=2.0))
print("Seed 1, temp=0.7, sans pénalité")
print(generate_once(seed=1, temperature=0.7, top_k=50, top_p=0.95))
print("="*60)

print("Sampling avec température basse (0.1):")
print(generate_once(seed=1, temperature=0.1, top_k=50, top_p=0.95))
print("Sampling avec température élevée (2.0):")
print(generate_once(seed=1, temperature=2.0, top_k=50, top_p=0.95))
print("="*60)

import time

print("Beam search num_beams=5:")
start = time.time()
out_beam5 = model.generate(
    **inputs,
    max_length=50,
    num_beams=5,
    early_stopping=True
)
txt_beam5 = tokenizer.decode(out_beam5[0], skip_special_tokens=True)
end = time.time()
print(txt_beam5)
print(f"Temps approximatif: {end-start:.2f}s")
print("-"*60)

print("Beam search num_beams=10:")
start = time.time()
out_beam10 = model.generate(
    **inputs,
    max_length=50,
    num_beams=10,
    early_stopping=True
)
txt_beam10 = tokenizer.decode(out_beam10[0], skip_special_tokens=True)
end = time.time()
print(txt_beam10)
print(f"Temps approximatif: {end-start:.2f}s")
print("-"*60)

print("Beam search num_beams=20:")
start = time.time()
out_beam20 = model.generate(
    **inputs,
    max_length=50,
    num_beams=20,
    early_stopping=True
)
txt_beam20 = tokenizer.decode(out_beam20[0], skip_special_tokens=True)
end = time.time()
print(txt_beam20)
print(f"Temps approximatif: {end-start:.2f}s")
