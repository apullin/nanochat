import torch
torch.backends.cuda.matmul.fp32_precision='tf32'
torch.backends.cudnn.conv.fp32_precision='tf32'
