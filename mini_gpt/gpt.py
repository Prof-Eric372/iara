from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F

#Hiperparâmetros
batch_size = 32
block_size = 8
n_embd = 32
n_cabecas = 4
n_camadas = 4
lr = 1e-3
max_passos = 5000
intervalo_eval = 500
iter_eval = 200
dropout = 0.0
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device)

CAMINHO = Path(__file__).parent/"data"/"domCasmurro.txt"

with open(CAMINHO, encoding="utf-8") as f:
    texto = f.read()

vocab = sorted(set(texto))
tamanho_vocab = len(vocab)
print("Tamanho do vocabulário:", tamanho_vocab)
#print("Caracteres: ", "".join(vocab))

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])
#TESTE CAPITU
#print(encode("Capitu"))
#print(decode(encode("Capitu")))

dados = torch.tensor(encode(texto), dtype=torch.long)
#print(dados.shape, dados.dtype)
#print(dados[:50])

n = int(0.9 * len(dados))
treino = dados[:n]
validacao = dados[n:]
print("Treino:", len(treino), "| Validação:", len(validacao))

#print(decode(dados[:50].tolist()))

#x = treino[:block_size]
#y = treino[1:block_size + 1]

#for t in range(block_size):
    #contexto = x[:t + 1]
    #alvo = y[t]
    #print(f"Entrada {contexto.tolist()} -> alvo {alvo.item()}")

torch.manual_seed(1337)


def get_batch(split):
    base = treino if split == "treino" else validacao
    ix = torch.randint(len(base) - block_size, (batch_size,))
    x = torch.stack([base[i:i + block_size] for i in ix])
    y = torch.stack([base[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


#xb, yb = get_batch("treino")
#print("x:", xb.shape)
#print(xb)
#print("y:", yb.shape)
#print(yb)


class Head(nn.Module):
    def __init__(self, tamanho_cabeca):
        super().__init__()
        self.key = nn.Linear(n_embd, tamanho_cabeca, bias=False)
        self.query = nn.Linear(n_embd, tamanho_cabeca, bias=False)
        self.value = nn.Linear(n_embd, tamanho_cabeca, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        notas = (q @ k.transpose(-2, -1)) * k.shape[-1] ** -0.5
        notas = notas.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        pesos = torch.softmax(notas, dim=-1)
        pesos = self.dropout(pesos)
        v = self.value(x)
        return pesos @ v

class MultiHead(nn.Module):
    def __init__(self, num_cabecas, tamanho_cabeca):
        super().__init__()
        self.cabecas = nn.ModuleList([Head(tamanho_cabeca) for _ in range(num_cabecas)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        saida = torch.cat([h(x) for h in self.cabecas], dim=-1)
        return self.dropout(self.proj(saida))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.rede(x)

class Bloco(nn.Module):
    def __init__(self):
        super().__init__()
        self.atencao = MultiHead(n_cabecas, n_embd // n_cabecas)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.atencao(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    
class ModeloAtencao(nn.Module):
    def __init__(self):
        super().__init__()
        self.tabela_tokens = nn.Embedding(tamanho_vocab, n_embd)
        self.tabela_posicoes = nn.Embedding(block_size, n_embd)
        self.blocos = nn.Sequential(*[Bloco() for _ in range(n_camadas)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, tamanho_vocab)

    def forward(self, idx, alvos=None):
        B, T = idx.shape
        emb_tokens = self.tabela_tokens(idx)
        emb_pos = self.tabela_posicoes(torch.arange(T, device=idx.device))
        x = emb_tokens + emb_pos
        x = self.blocos(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
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
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            prox = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, prox), dim=1)
        return idx


modelo = ModeloAtencao().to(device)
print(sum(p.numel() for p in modelo.parameters()), "parâmetros")
xb, yb = get_batch("treino")
logits, loss = modelo(xb, yb)
print(logits.shape, loss.item())

@torch.no_grad()
def estimar_loss():
    modelo.eval()
    saida = {}
    for split in ["treino", "validacao"]:
        losses = torch.zeros(iter_eval)
        for k in range(iter_eval):
            xb, yb = get_batch(split)
            _, loss = modelo(xb, yb)
            losses[k] = loss.item()
        saida[split] = losses.mean().item()
    modelo.train()
    return saida


optimizer = torch.optim.AdamW(modelo.parameters(), lr=lr)

for passo in range(max_passos):
    if passo % intervalo_eval == 0:
        perdas = estimar_loss()
        print(f"passo {passo}: treino {perdas['treino']:.4f} | validação {perdas['validacao']:.4f}")
    xb, yb = get_batch("treino")
    logits, loss = modelo(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

modelo.eval()
inicio = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(modelo.generate(inicio, 300)[0].tolist()))

torch.save({
    "pesos": modelo.state_dict(),
    "vocab": vocab,
    "config": {"n_embd": n_embd, "n_cabecas": n_cabecas,
               "n_camadas": n_camadas, "block_size": block_size},
}, Path(__file__).parent / "minigpt.pt")
print("Modelo salvo em minigpt.pt")