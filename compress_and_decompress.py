import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from models import TCM  # 确保这个import能正确导入你的TCM模型
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Compress and decompress images using a trained model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--input_dir", type=str, default="/root/autodl-tmp/AerialImageDataset/train/images_split", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="compressed", help="Directory to save compressed images")
    parser.add_argument("--N", type=int, default=128, help="N")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
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

def main():
    args = parse_args()
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)
    net = net.to(device)
    net.eval()
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()})
    
    # 更新模型（如果需要）
    net.update()
    
    # 处理每张图片

    for root, _, files in os.walk(args.input_dir):

        for filename in files:

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, filename)

                relative_path = os.path.relpath(root, args.input_dir)

                # Separate directories for binary and decompressed images
                output_bin_dir = os.path.join(args.output_dir, "bin", relative_path)

                output_img_dir = os.path.join(args.output_dir, "images", relative_path)

                output_bin_path = os.path.join(output_bin_dir, f"{os.path.splitext(filename)[0]}.bin")

                output_json_path = os.path.join(output_bin_dir, f"{os.path.splitext(filename)[0]}.json")
                output_img_path = os.path.join(output_img_dir, f"{os.path.splitext(filename)[0]}_decompressed.jpg")

                

                # 创建输出子目录

                os.makedirs(output_bin_dir, exist_ok=True)

                os.makedirs(output_img_dir, exist_ok=True)
                

                img = Image.open(input_path).convert('RGB')

                x = transforms.ToTensor()(img).unsqueeze(0).to(device)

                

                # 填充图像

                x_padded, padding = pad(x, args.N)
                

                with torch.no_grad():

                    out_enc = net.compress(x_padded)
                

                # 保存压缩后的二进制数据
                with open(output_bin_path, 'wb') as f:
                    for s in out_enc["strings"]:
                        f.write(s[0])
                

                # 保存额外信息到JSON文件，包括每个字符串的长度

                extra_info = {
                    "shape": out_enc["shape"],

                    "padding": padding,

                    "string_lengths": [len(s[0]) for s in out_enc["strings"]]
                }

                

                with open(output_json_path, 'w') as f:
                    json.dump(extra_info, f)
                
                print(f"Compressed {input_path} and saved to {output_bin_path} and {output_json_path}")

                
                # 解压缩图像

                with torch.no_grad():
                    out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                

                # 去除填充
                x_decompressed = crop(out_dec["x_hat"], padding)

                

                # 保存解压后的图像

                img_decompressed = transforms.ToPILImage()(x_decompressed.squeeze().cpu())

                img_decompressed.save(output_img_path)
                
                print(f"Decompressed {input_path} and saved to {output_img_path}")




if __name__ == "__main__":
    main()