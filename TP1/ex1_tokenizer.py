from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Phrase 1
phrase = "Artificial intelligence is metamorphosing the world!"
tokens = tokenizer.tokenize(phrase)
print("Tokens:")
print(tokens)
token_ids = tokenizer.encode(phrase)
print("\nToken IDs:")
print(token_ids)
print("\nDétails par token:")
for tid in token_ids:
    txt = tokenizer.decode([tid])
    print(tid, repr(txt))

# Phrase 2
phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."
tokens2 = tokenizer.tokenize(phrase2)
print("\nTokens phrase 2:")
print(tokens2)

# Extraction correcte des sous-tokens
start = tokens2.index("Ġant")
end = tokens2.index("ism") + 1
long_word_tokens = tokens2[start:end]
print("\nSous-tokens du mot 'antidisestablishmentarianism':")
print(long_word_tokens)
print("Nombre de sous-tokens:", len(long_word_tokens))