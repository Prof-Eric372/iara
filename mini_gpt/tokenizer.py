from pathlib import Path

CAMINHO = Path(__file__).parent/"data"/"domCasmurro.txt"

with open(CAMINHO, encoding="utf-8") as f:
    texto = f.read()

vocab = sorted(set(texto))
tamanho_vocab = len(vocab)
print("Tamanho do vocabulário:", tamanho_vocab)
print("Caracteres: ", "".join(vocab))

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])

print(encode("Capitu"))
print(decode(encode("Capitu")))