import torch

torch.manual_seed(42)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# Versão lenta: média dos anteriores com laços
xmedia = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        anteriores = x[b, :t + 1]
        xmedia[b, t] = anteriores.mean(dim=0)

# Versão com matriz triangular
pesos = torch.tril(torch.ones(T, T))
pesos = pesos / pesos.sum(dim=1, keepdim=True)
xmedia2 = pesos @ x

print(torch.allclose(xmedia, xmedia2))
print(pesos)

notas = torch.zeros(T, T)
notas = notas.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
pesos3 = torch.softmax(notas, dim=-1)
xmedia3 = pesos3 @ x

print(torch.allclose(xmedia, xmedia3))
print(pesos3)