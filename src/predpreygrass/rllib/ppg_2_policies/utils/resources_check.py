import torch
import multiprocessing
import GPUtil
import psutil

mem = psutil.virtual_memory()
total_ram_gb = mem.total / 1e9  # Convert bytes to GB

print(f"Total system RAM: {total_ram_gb:.2f} GB")
# CUDA status and GPU name
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU only"

print(f"CUDA available: {cuda_available}")
print(f"CUDA device: {device_name}")

# CPU and GPU info
num_cpus = multiprocessing.cpu_count()
gpus = GPUtil.getGPUs()
num_gpus = len(gpus)
gpu_names = [gpu.name for gpu in gpus]

print(f"Detected {num_cpus} CPU cores")
print(f"Detected {num_gpus} GPU(s): {gpu_names}")
