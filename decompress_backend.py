import sys
import os
from PyQt5.QtCore import QThread, pyqtSignal
import torch
from decompress_images import load_model, decompress_bin_file, assemble_grid
import heapq
from math import sqrt
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
import concurrent.futures
import threading  # 引入 threading 模块

class ImageLoaderThread(QThread):
    # 发送单个图像及其位置的信号
    progress_image = pyqtSignal(QPixmap, tuple)  # pixmap, position

    def __init__(self, net, bin_files, bin_path, grid_size, device, batch_size=8, max_workers=4):
        super().__init__()
        self.net = net
        self.bin_files = bin_files
        self.bin_path = bin_path
        self.grid_size = grid_size
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._is_running = True
        self.center_position = (0, 0)
        self.image_queue = []
        self.loaded_images = set()
        self.images = []
        self.lock = threading.Lock()  # 初始化锁

    def set_center_position(self, position):
        # 输出当前位置
        print(f"Setting center position to: {position}")
        self.center_position = position
        self._update_queue()

    def _update_queue(self):
        # Clear the current queue
        self.image_queue = []
        # Recalculate distances and update the priority queue based on the new center position
        for i, j, bin_path in self.bin_files:
            if bin_path not in self.loaded_images:
                distance = sqrt(
                    (i * 256 - self.center_position[0])**2 + 
                    (j * 256 - self.center_position[1])**2
                )
                heapq.heappush(self.image_queue, (distance, (i, j, bin_path)))
    

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            while self.image_queue and self._is_running:
                # 提交任务直到达到最大工作线程数
                while len(futures) < self.max_workers and self.image_queue and self._is_running:
                    _, (i, j, bin_path) = heapq.heappop(self.image_queue)
                    
                    with self.lock:
                        if bin_path in self.loaded_images:
                            continue
                        self.loaded_images.add(bin_path)  # 提交时标记为已加载

                    json_filename = f"{os.path.splitext(os.path.basename(bin_path))[0]}.json"
                    json_path = os.path.join(self.bin_path, json_filename)
                    if not os.path.exists(json_path):
                        print(f"JSON file {json_path} not found for bin file {bin_path}. Skipping.")
                        continue

                    # 提交任务并记录相关信息
                    future = executor.submit(decompress_bin_file, self.net, bin_path, json_path, self.device)
                    futures[future] = (i, j, bin_path)

                if not futures:
                    break

                # 等待任意一个任务完成
                done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)

                for future in done:
                    i, j, bin_path = futures.pop(future)
                    try:
                        x_hat = future.result()
                        with self.lock:
                            self.images.append({'position': (i, j), 'image': x_hat.squeeze(0)})
                    except Exception as e:
                        print(f"Error decompressing {bin_path}: {e}")
                        with self.lock:
                            self.loaded_images.discard(bin_path)  # 失败时移除标记

                    # 直接发送单个图像及其位置
                    if self.images:
                        image_data = self.images.pop(0)  # 获取并移除第一个元素
                        grid_image = image_data['image']
                        position = image_data['position']
                        pixmap = self._convert_to_pixmap(grid_image)
                        self.progress_image.emit(pixmap, position)

            # 处理所有剩余的完成任务
            for future in concurrent.futures.as_completed(futures):
                i, j, bin_path = futures.pop(future)
                try:
                    x_hat = future.result()
                    with self.lock:
                        self.images.append({'position': (i, j), 'image': x_hat.squeeze(0)})
                except Exception as e:
                    print(f"Error decompressing {bin_path}: {e}")
                    with self.lock:
                        self.loaded_images.discard(bin_path)

                if self.images:
                    image_data = self.images.pop(0)
                    grid_image = image_data['image']
                    position = image_data['position']
                    pixmap = self._convert_to_pixmap(grid_image)
                    self.progress_image.emit(pixmap, position)

        # 最终发送已加载的图像
        while self.images and self._is_running:
            image_data = self.images.pop(0)
            grid_image = image_data['image']
            position = image_data['position']
            pixmap = self._convert_to_pixmap(grid_image)
            self.progress_image.emit(pixmap, position)

    def _convert_to_pixmap(self, grid_image):
        """
        将 PyTorch 张量转换为 QPixmap。
        """
        # 将张量移动到 CPU 并转换为 NumPy 数组
        grid_image_np = grid_image.cpu().numpy().transpose(1, 2, 0)  # 从 (C, H, W) 到 (H, W, C)

        # 确保图像数据在 [0, 255] 范围内，并转换为 uint8 类型
        grid_image_np = np.clip(grid_image_np * 255, 0, 255).astype(np.uint8)

        height, width, channel = grid_image_np.shape
        bytes_per_line = 3 * width  # 对于 RGB888，每行字节数为 3 * 宽度

        # 将 NumPy 数组转换为 QImage
        qimage = QImage(
            grid_image_np.tobytes(),  # 确保数据类型为 bytes
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        # 将 QImage 转换为 QPixmap
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def stop(self):
        self._is_running = False