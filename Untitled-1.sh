# 训练模型
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python -u ./train.py -d /root/autodl-tmp/AerialImageDataset/train/images_split \
    --cuda --N 128 --lambda 0.05 --epochs 20 --lr_epoch 15 18 \
    --save_path ./save/ --save' > outputlog.txt 2>&1 &

# 压缩图片
python compress_images.py --checkpoint save/0.05checkpoint_best.pth.tar --cuda

# 压缩图片并打包
python compressToTar.py --input_dir ./LIC_TCM/output --output_zip ./LIC_TCM/output/output.tar