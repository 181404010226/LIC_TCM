import os
import shutil

# 定义源文件夹和目标文件夹
source_folder = 'compressedBIN'
target_base_folder = 'classified_BIN'

# 确保源文件夹存在
if not os.path.exists(source_folder):
    print(f"错误: 源文件夹 '{source_folder}' 不存在。")
    exit(1)

# 创建目标基础文件夹（如果不存在）
if not os.path.exists(target_base_folder):
    os.makedirs(target_base_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.bin') or filename.endswith('.json'):
        # 获取文件前缀（假设前缀在第一个下划线之前）
        prefix = filename.split('_')[0]
        
        # 创建对应前缀的目标文件夹（如果不存在）
        target_folder = os.path.join(target_base_folder, prefix)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # 移动文件到对应的目标文件夹
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder, filename)
        shutil.move(source_file, target_file)
        print(f"已移动 {filename} 到 {target_folder}")

print("文件分类完成。")