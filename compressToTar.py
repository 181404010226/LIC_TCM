import os
import zipfile
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Pack all .bin files in a directory into a zip file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .bin files")
    parser.add_argument("--output_zip", type=str, default="packed_bin_files.zip", help="Name of the output zip file")
    return parser.parse_args()

def pack_bin_files(input_dir, output_zip):
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # 创建一个ZipFile对象
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历输入目录
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.bin') or file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    # 将文件添加到zip文件中，使用相对路径
                    arcname = os.path.relpath(file_path, input_dir)
                    zipf.write(file_path, arcname)
                    print(f"Added {file} to {output_zip}")

    print(f"All .bin files have been packed into {output_zip}")

def main():
    args = parse_args()
    pack_bin_files(args.input_dir, args.output_zip)

if __name__ == "__main__":
    main()