from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F

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


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tabela = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, alvos=None):
        logits = self.tabela(idx)
        if alvos is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            alvos = alvos.view(B * T)
            loss = F.cross_entropy(logits, alvos)
        return logits, loss

    def generate(self, idx, max_novos):
        for _ in range(max_novos):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            prox = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, prox), dim=1)
        return idx

modelo = BigramModel(tamanho_vocab)
logits, loss = modelo(xb, yb)
print(logits.shape, loss.item())
inicio = torch.zeros((1, 1), dtype=torch.long)
print(decode(modelo.generate(inicio, 100)[0].tolist()))

optimizer = torch.optim.AdamW(modelo.parameters(), lr=1e-2)

batch_size = 32
for passo in range(10000):
    xb, yb = get_batch("treino")
    logits, loss = modelo(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if passo % 1000 == 0:
        print(passo, loss.item())

print("loss final:", loss.item())
print(decode(modelo.generate(inicio, 300)[0].tolist()))