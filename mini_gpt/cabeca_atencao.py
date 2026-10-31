import torch
import torch.nn as nn

torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

tamanho_cabeca = 16
key = nn.Linear(C, tamanho_cabeca, bias=False)
query = nn.Linear(C, tamanho_cabeca, bias=False)
value = nn.Linear(C, tamanho_cabeca, bias=False)

k = key(x)
q = query(x)
notas = (q @ k.transpose(-2, -1)) * tamanho_cabeca ** -0.5

tril = torch.tril(torch.ones(T, T))
notas = notas.masked_fill(tril == 0, float("-inf"))
pesos = torch.softmax(notas, dim=-1)

v = value(x)
saida = pesos @ v

print(saida.shape)
print(pesos[0])