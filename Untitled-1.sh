# 训练模型
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python -u ./train.py -d /root/autodl-tmp/AerialImageDataset/train/images_split \
    --cuda --N 128 --lambda 0.05 --epochs 20 --lr_epoch 15 18 \
    --save_path ./save/ --save' > outputlog.txt 2>&1 &

# 压缩图片
python compress_images.py --checkpoint save/0.05checkpoint_best.pth.tar --cuda --input_dir output --output_dir compressedBIN --N 64

# 压缩解压图片
python compress_and_decompress.py --checkpoint 3dtilesave/0.05checkpoint_latest.pth.tar --cuda --input_dir cropped_images --output_dir 3dtilesCompressedBIN --N 64


# 压缩图片并打包
python compressToTar.py --input_dir compressed --output_zip output.tar

# 解压图片
python decompress_images.py --prefix austin10 --bin_path compressedBIN --checkpoint save/0.05checkpoint_best.pth.tar --output output_image.png --cuda --N 64

# 评估
python eval.py --model save/0.05checkpoint_best.pth.tar --image output\austin1_0_0.png --output output_image_restored.png --cuda

#windows
$env:CUDA_VISIBLE_DEVICES=0
python -u train.py -d D:\VisutalStudio\repository\LIC_TCM\output `
--cuda --N 64 --lambda 0.05 --epochs 10 --lr_epoch 5 8 `
--save_path D:\VisutalStudio\repository\LIC_TCM\save --save
