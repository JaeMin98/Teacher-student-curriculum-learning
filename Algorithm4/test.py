import torch

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.version())