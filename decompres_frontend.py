import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QComboBox, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
import torch
from decompress_backend import ImageLoaderThread  # Importing from backend

from decompress_images import load_model  # Assuming decompress_images is part of backend dependencies

class SatelliteImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.net = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.grid_size = 19
        self.bin_path = "compressedBIN"  # Directory containing bin files
        self.checkpoint = "save/0.05checkpoint_best.pth.tar"  # Model checkpoint file
        self.N = 64
        self.scale_factor = 1.0  # Scale factor for zooming
        self.last_mouse_position = QPoint()
        self.thread = None  # Reference to the current thread
        self.center_position = (0, 0)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Create prefix dropdown and label
        input_layout = QHBoxLayout()
        self.prefix_combo = QComboBox()
        self.load_label = QLabel("请选择前缀")
        self.load_prefixes()
        self.prefix_combo.currentIndexChanged.connect(self.load_image)
        input_layout.addWidget(self.load_label)
        input_layout.addWidget(self.prefix_combo)
        layout.addLayout(input_layout)

        # Create QGraphicsView and QGraphicsScene to display images
        self.graphics_view = QGraphicsView()
        self.graphics_view.setAlignment(Qt.AlignCenter)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        layout.addWidget(self.graphics_view)

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
        # Stop existing thread if running
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()

        prefix = self.prefix_combo.currentText()
        if not prefix:
            return

        if self.net is None:
            self.net = load_model(self.checkpoint, self.device, self.N)

        # Collect bin files
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

        # Sort bin files
        bin_files.sort()

        # Clear existing scene
        self.scene.clear()

        # Start the image loading thread
        self.thread = ImageLoaderThread(self.net, bin_files, self.bin_path, self.grid_size, self.device)
        self.thread.progress_image.connect(self.update_image)
        self.thread.set_center_position(self.center_position)
        self.thread.start()

    def update_image(self, pixmap, position):
        """
        接收单个图像及其位置，并将其添加到场景中。
        """
        # 计算图像在场景中的位置
        x, y = position
        item = QGraphicsPixmapItem(pixmap)
        item.setPos(x * 256, y * 256)  # 假设每张图片的尺寸为256x256
        self.scene.addItem(item)

    def refresh_image(self):
        # 不再需要，因为每个图像独立添加
        pass

    def update_center_position(self):
        # 可以根据需要实现，当前不需要
        pass

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.thread and self.thread.isRunning():
            delta = event.globalPos() - self.last_mouse_position
            self.last_mouse_position = event.globalPos()
            self.graphics_view.horizontalScrollBar().setValue(
                self.graphics_view.horizontalScrollBar().value() - delta.x()
            )
            self.graphics_view.verticalScrollBar().setValue(
                self.graphics_view.verticalScrollBar().value() - delta.y()
            )
            # self.update_center_position()  # 根据需要调整

    def wheelEvent(self, event):
        if not self.thread or not self.thread.isRunning():
            return
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
        self.scale_factor *= factor
        self.scale_factor = max(0.1, min(self.scale_factor, 5))
        self.graphics_view.resetTransform()
        self.graphics_view.scale(self.scale_factor, self.scale_factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_position = event.globalPos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_position = QPoint()

    def closeEvent(self, event):
        # Ensure the thread is properly stopped before closing
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        event.accept()

if __name__ == '__main__':
        app = QApplication(sys.argv)
        ex = SatelliteImageViewer()
        sys.exit(app.exec_())