import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(f'Current CUDA device: {torch.cuda.get_device_name(device)}')
props = torch.cuda.get_device_properties(device)
print(f'Memory available on device: {props.total_memory / 1024**2:.2f} MB')