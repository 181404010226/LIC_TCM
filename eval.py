import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings
import os
import sys
import math
import argparse
import time
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


warnings.filterwarnings("ignore")


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.).item())


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()


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


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Image Compression and Restoration Script.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for computation")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="Gradient clipping max norm (default: %(default)s)",
    )
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--output", type=str, default="restored_image.png", help="Path to save the restored image"
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    p = 128

    device = 'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the model
    net = TCM(config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8],
              drop_path_rate=0.0, N=64, M=320)
    net = net.to(device)
    net.eval()

    # Load the checkpoint
    if args.model:
        print(f"Loading model from {args.model}")
        checkpoint = torch.load(args.model, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        net.load_state_dict(state_dict)
    
    net.update()

    # Load and preprocess the image
    img = transforms.ToTensor()(Image.open(args.image).convert('RGB')).to(device)
    x = img.unsqueeze(0)
    x_padded, padding = pad(x, p)

    with torch.no_grad():
        if args.cuda:
            torch.cuda.synchronize()
        start_time = time.time()
        out_enc = net.compress(x_padded)
        out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
        if args.cuda:
            torch.cuda.synchronize()
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds

    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
    out_dec["x_hat"].clamp_(0, 1)

    # Save the restored image
    restored_img = transforms.ToPILImage()(out_dec["x_hat"].squeeze(0).cpu())
    restored_img.save(args.output)
    print(f"Restored image saved to {args.output}")

    # Compute metrics
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    psnr = compute_psnr(x, out_dec["x_hat"])
    ms_ssim_val = compute_msssim(x, out_dec["x_hat"])

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image
    ax1.imshow(img.permute(1, 2, 0).cpu())
    ax1.axis('off')
    ax1.set_title('Original')

    # Plot restored image
    ax2.imshow(out_dec["x_hat"].squeeze(0).permute(1, 2, 0).cpu())
    ax2.axis('off')
    ax2.set_title('Restored')

    # Add metrics text
    plt.figtext(0.5, 0.01, f'Ours [MSE]\n{bpp:.3f}bpp|{psnr:.2f}PSNR-dB|{ms_ssim_val:.2f}SSIM-dB', 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Save the figure
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Result image saved to {args.output}")

    # Print metrics (optional, you can remove if not needed)
    print(f'Bit-rate: {bpp:.3f} bpp')
    print(f'MS-SSIM: {ms_ssim_val:.4f} dB')
    print(f'PSNR: {psnr:.2f} dB')
    print(f'Compression and restoration time: {total_time:.3f} ms')



if __name__ == "__main__":
    main(sys.argv[1:])