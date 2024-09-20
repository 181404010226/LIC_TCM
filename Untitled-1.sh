# 训练模型
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python -u ./train.py -d /root/autodl-tmp/AerialImageDataset/train/images_split \
    --cuda --N 128 --lambda 0.05 --epochs 20 --lr_epoch 15 18 \
    --save_path ./save/ --save' > outputlog.txt 2>&1 &

# 压缩图片
python compress_images.py --checkpoint save/0.05checkpoint_best.pth.tar --cuda --input_dir output --output_dir compressedBIN --N 64


# 压缩图片并打包
python compressToTar.py --input_dir compressed --output_zip output.tar

# 解压图片
python decompress_images.py --prefix austin1 --bin_path compressed --checkpoint save/0.05checkpoint_best.pth.tar --output output_image.png --cuda --N 64

#windows
$env:CUDA_VISIBLE_DEVICES=0
python -u train.py -d D:\visualStudioResposity\LIC_TCM\output `
--cuda --N 64 --lambda 0.05 --epochs 10 --lr_epoch 5 8 `
    --save_path D:\visualStudioResposity\LIC_TCM\save --save
