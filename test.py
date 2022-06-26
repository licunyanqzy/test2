import torch.nn as nn
import torch

softmax = nn.Softmax(dim=0)
a = torch.tensor([0.1, 0.3, 0.2, 0.5, 0.1, 0, 0, 0, 0, 0])
b = softmax(a)
print(b)

neg_inf = (-1e8) * torch.ones_like(a)
aa = torch.where(a == 0, neg_inf, a)
bb = softmax(aa)
print(aa)
print(bb)

