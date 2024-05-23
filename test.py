import torch
from main import VPRModel
vpr = VPRModel(agg_arch='gem', agg_config={'p': 3})
x = torch.randn(4, 3, 320, 320)
y = vpr.forward(x)
print(y.shape)
