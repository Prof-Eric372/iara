from pathlib import Path
import torch

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

dados = torch.tensor(encode(texto), dtype=torch.long)
print(dados.shape, dados.dtype)
print(dados[:50])

n = int(0.9 * len(dados))
treino = dados[:n]
validacao = dados[n:]
print("Treino:", len(treino), "| Validação:", len(validacao))

print(decode(dados[:50].tolist()))

block_size = 8
x = treino[:block_size]
y = treino[1:block_size + 1]

for t in range(block_size):
    contexto = x[:t + 1]
    alvo = y[t]
    print(f"Entrada {contexto.tolist()} -> alvo {alvo.item()}")

torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batch(split):
    base = treino if split == "treino" else validacao
    ix = torch.randint(len(base) - block_size, (batch_size,))
    x = torch.stack([base[i:i + block_size] for i in ix])
    y = torch.stack([base[i + 1:i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("treino")
print("x:", xb.shape)
print(xb)
print("y:", yb.shape)
print(yb)