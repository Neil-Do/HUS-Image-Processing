import torch
import numpy as np
from torchvision import transforms
A = np.ones((100, 3,3))
# print(A)
B = torch.tensor(A)
print(B)
C = torch.unsqueeze(B, 1)
print(C.shape)
print(B.shape)
