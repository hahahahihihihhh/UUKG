import torch.nn as nn
import torch
from statsmodels.sandbox.panel.sandwich_covariance_generic import kernel

input = torch.randn(5, 2, 20, 50)
m1d = nn.Conv1d(2, 3, kernel_size=(3, 2))
output1 = m1d(input)
print(output1.shape)



# m2d = nn.Conv2d(2, 3, kernel_size=(1, 1))
#
# output2 = m2d(input)

# print(output1, 111, output2)