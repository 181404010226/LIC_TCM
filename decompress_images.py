import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from models import TCM  # Ensure this imports your TCM model correctly
import json
import math
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Decompress bin files and assemble into a PNG image.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix of the bin files (e.g., vienna7)")
    parser.add_argument("--bin_path", type=str, required=True, help="Path to the directory containing bin files")
    parser.add_argument("--output", type=str, default="assembled_image.png", help="Output PNG file name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--N", type=int, default=128, help="N")
    parser.add_argument("--grid_size", type=int, default=19, help="Grid size (e.g., 18 for 18x18)")
    return parser.parse_args()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def load_model(checkpoint_path, device):
    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)
    net = net.to(device)
    net.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    net.load_state_dict(state_dict)
    net.update()
    return net

def decompress_bin_file(net, bin_file, json_file, device):
    # 读取JSON文件以获取字符串长度和其他信息
    with open(json_file, 'r') as f:
        extra_info = json.load(f)
    
    shape = extra_info["shape"]
    padding = extra_info["padding"]
    string_lengths = extra_info.get("string_lengths", [])
    
    if not string_lengths:
        print(f"No string_lengths found in {json_file}.")
        sys.exit(1)
    
    # 读取二进制文件并根据字符串长度分割
    strings = []
    with open(bin_file, 'rb') as f:
        for length in string_lengths:
            s = f.read(length)
            if not s:
                break
            strings.append([s])
    
    # 检查是否正确读取所有字符串
    if len(strings) != len(string_lengths):
        print(f"Expected {len(string_lengths)} strings, but got {len(strings)} in {bin_file}.")
        sys.exit(1)
    
    # 解压缩
    with torch.no_grad():
        out_dec = net.decompress(strings, shape)
    
    x_hat = out_dec["x_hat"].clamp_(0, 1)
    x_hat = crop(x_hat, padding)
    return x_hat

def assemble_grid(images, grid_size):
    if not images:
        raise ValueError("No images to assemble.")
    
    # Assume all images have the same size
    C, H, W = images[0]['image'].shape
    grid_image = torch.zeros((C, H * grid_size, W * grid_size))
    
    for img in images:
        i, j = img['position']
        grid_image[:, i*H:(i+1)*H, j*W:(j+1)*W] = img['image']
    
    grid_image = grid_image.permute(1, 2, 0).cpu().numpy() * 255
    grid_image = grid_image.astype('uint8')
    return Image.fromarray(grid_image)

def main():
    args = parse_args()
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    # Load model
    print("Loading model...")
    net = load_model(args.checkpoint, device)
    
    # Collect bin files
    bin_files = []
    for filename in os.listdir(args.bin_path):
        parts = filename.split('_')
        if len(parts) >= 3 and '_'.join(parts[:-2]) == args.prefix and filename.endswith(".bin"):
            try:
                i = int(parts[-2])
                j = int(parts[-1].replace(".bin", ""))
                bin_files.append((i, j, os.path.join(args.bin_path, filename)))
            except ValueError:
                continue
    
    if not bin_files:
        print(f"No bin files found with prefix {args.prefix} in {args.bin_path}")
        sys.exit(1)
    
    # Sort bin files based on i and j
    bin_files.sort()
    
    grid_size = args.grid_size
    expected_files = grid_size * grid_size
    if len(bin_files) != expected_files:
        print(f"Expected {expected_files} bin files for a {grid_size}x{grid_size} grid, but found {len(bin_files)}.")
        sys.exit(1)

    images = []
    for i, j, bin_path in bin_files:
        json_filename = f"{os.path.splitext(os.path.basename(bin_path))[0]}.json"
        json_path = os.path.join(args.bin_path, json_filename)
        if not os.path.exists(json_path):
            print(f"JSON file {json_path} not found for bin file {bin_path}. Skipping.")
            continue
        x_hat = decompress_bin_file(net, bin_path, json_path, device)
        images.append({'position': (j, i), 'image': x_hat.squeeze(0)})
        print(f"Decompressed {bin_path}")
    
    # Check if all images are collected
    if len(images) != expected_files:
        print(f"Collected {len(images)} images, expected {expected_files}.")
        sys.exit(1)
    
    # Assemble grid
    print("Assembling images into a grid...")
    grid_image = assemble_grid(images, grid_size)
    
    # Save as PNG
    grid_image.save(args.output)
    print(f"Saved assembled image to {args.output}")

if __name__ == "__main__":
    main()