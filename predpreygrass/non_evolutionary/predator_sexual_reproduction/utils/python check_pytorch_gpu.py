# check_pytorch_gpu.py
import torch

print("🔍 PyTorch version:", torch.__version__)
print("🧠 CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🖥️  GPU name:", torch.cuda.get_device_name(0))
    print("🧱 GPU compute capability:", torch.cuda.get_device_capability(0))
    print("⚙️  CUDA version (runtime):", torch.version.cuda)
    print("🧮 Supported archs:", torch.cuda.get_arch_list())
else:
    print("🚫 No GPU detected by PyTorch.")
