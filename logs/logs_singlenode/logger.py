import torch

print("PyTorch version:", torch.__version__)
print("CUDA version used by PyTorch:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
