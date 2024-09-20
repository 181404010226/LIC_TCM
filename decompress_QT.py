import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QComboBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
import torch
from PIL import Image
import numpy as np
from decompress_images import load_model, decompress_bin_file, assemble_grid


class ImageLoaderThread(QThread):
    progress = pyqtSignal(QPixmap)

    def __init__(self, net, bin_files, bin_path, grid_size, device):
        super().__init__()
        self.net = net
        self.bin_files = bin_files
        self.bin_path = bin_path
        self.grid_size = grid_size
        self.device = device
        self._is_running = True  # 添加运行标志

    def run(self):
        images = []
        for i, j, bin_path in self.bin_files:
            if not self._is_running:
                print("ImageLoaderThread 已被停止。")
                break  # 检查是否停止线程

            json_filename = f"{os.path.splitext(os.path.basename(bin_path))[0]}.json"
            json_path = os.path.join(self.bin_path, json_filename)
            if not os.path.exists(json_path):
                print(f"JSON file {json_path} not found for bin file {bin_path}. Skipping.")
                continue
            try:
                x_hat = decompress_bin_file(self.net, bin_path, json_path, self.device)
            except Exception as e:
                print(f"Error decompressing {bin_path}: {e}")
                continue
            images.append({'position': (j, i), 'image': x_hat.squeeze(0)})

            # 组装当前已加载的图像网格
            grid_image = assemble_grid(images, self.grid_size)

            # 转换为 QPixmap 并发送信号
            grid_image_np = np.array(grid_image)
            height, width, channel = grid_image_np.shape
            bytes_per_line = 3 * width
            qimage = QImage(grid_image_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.progress.emit(pixmap)

        # 加载完成后发送最终的图像
        if images and self._is_running:
            self.progress.emit(pixmap)

    def stop(self):
        self._is_running = False  # 设置运行标志为 False


class SatelliteImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.net = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.grid_size = 19
        self.bin_path = "compressedBIN"  # 指定存放 bin 文件的文件夹
        self.checkpoint = "save/0.05checkpoint_best.pth.tar"  # 指定模型 checkpoint 文件
        self.N = 64
        self.scale_factor = 1.0  # 缩放因子
        self.pixmap = None
        self.last_mouse_position = QPoint()
        self.thread = None  # 当前的线程引用
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 创建前缀下拉列表和加载标签
        input_layout = QHBoxLayout()
        self.prefix_combo = QComboBox()
        self.load_label = QLabel("请选择前缀")
        self.load_prefixes()
        self.prefix_combo.currentIndexChanged.connect(self.load_image)
        input_layout.addWidget(self.load_label)
        input_layout.addWidget(self.prefix_combo)
        layout.addLayout(input_layout)

        # 创建滚动区域来显示图像
        self.scroll_area = QScrollArea()
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setBackgroundRole(self.scroll_area.backgroundRole())
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(True)
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)
        self.setGeometry(100, 100, 1024, 768)
        self.setWindowTitle('Satellite Image Viewer')
        self.show()

    def load_prefixes(self):
        prefixes = set()
        for filename in os.listdir(self.bin_path):
            if filename.endswith('.bin'):
                parts = filename.split('_')
                if len(parts) >= 3:
                    prefix = '_'.join(parts[:-2])
                    prefixes.add(prefix)
        self.prefix_combo.addItems(sorted(prefixes))

    def load_image(self):
        # 如果已有线程正在运行，先停止它
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()

        prefix = self.prefix_combo.currentText()
        if not prefix:
            return

        if self.net is None:
            self.net = load_model(self.checkpoint, self.device, self.N)

        # 收集 bin 文件
        bin_files = []
        for filename in os.listdir(self.bin_path):
            parts = filename.split('_')
            if len(parts) >= 3 and '_'.join(parts[:-2]) == prefix and filename.endswith(".bin"):
                try:
                    i = int(parts[-2])
                    j = int(parts[-1].replace(".bin", ""))
                    bin_files.append((i, j, os.path.join(self.bin_path, filename)))
                except ValueError:
                    continue

        if not bin_files:
            print(f"No bin files found with prefix {prefix} in {self.bin_path}")
            return

        # 排序 bin 文件
        bin_files.sort()

        # 启动图像加载线程
        self.thread = ImageLoaderThread(self.net, bin_files, self.bin_path, self.grid_size, self.device)
        self.thread.progress.connect(self.update_image)
        self.thread.start()

    def update_image(self, pixmap):
        self.pixmap = pixmap
        self.refresh_image()

    def refresh_image(self):
        if self.pixmap:
            scaled_pixmap = self.pixmap.scaled(
                self.pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())

    def wheelEvent(self, event):
        if not self.pixmap:
            return
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
        self.scale_factor *= factor
        self.scale_factor = max(0.1, min(self.scale_factor, 5))
        self.refresh_image()
        event.accept()  # 阻止事件进一步传播

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_position = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.pixmap:
            delta = event.globalPos() - self.last_mouse_position
            self.last_mouse_position = event.globalPos()
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x())
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_position = QPoint()

    def closeEvent(self, event):
        # 确保线程在关闭应用前被干净地停止
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SatelliteImageViewer()
    sys.exit(app.exec_())