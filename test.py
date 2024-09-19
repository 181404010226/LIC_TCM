import torch
import sys

def print_versions():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    else:
        print("CUDA is not available")

if __name__ == "__main__":
    print_versions()
