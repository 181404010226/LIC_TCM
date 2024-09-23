import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QComboBox, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtGui import QPixmap, QImage, QBrush, QColor # Import QBrush and QColor
from PyQt5.QtCore import Qt, QPoint, QRectF, QEvent  # 添加 QEvent 导入
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
        self.last_mouse_position = QPoint()
        self.is_right_mouse_pressed = False
        self.scale_factor = 1.0
        self.thread = None  # Reference to the current thread
        self.center_position = (19*128, 19*128)
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

        # 添加使用提示标签
        usage_tip = QLabel("使用说明：鼠标左键拖动移动图像，鼠标右键左右拖动进行缩放")
        usage_tip.setAlignment(Qt.AlignCenter)
        usage_tip.setStyleSheet("color: black; font-family: SimSun, serif; font-size: 14px;")
        layout.addWidget(usage_tip)

        # Create QGraphicsView and QGraphicsScene to display images
        self.graphics_view = QGraphicsView()
        self.graphics_view.setAlignment(Qt.AlignCenter)
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        self.graphics_view.viewport().installEventFilter(self)

        self.scene = QGraphicsScene()

        # Set the scene background to black
        self.scene.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.refresh_black_canvas()

        self.graphics_view.setScene(self.scene)
        layout.addWidget(self.graphics_view)
        self.setLayout(layout)
        self.setGeometry(100, 100, 1024, 768)
        self.setWindowTitle('Satellite Image Viewer')
        self.show()

    # 添加这个新方法
    def eventFilter(self, source, event):
        if source == self.graphics_view.viewport():
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.last_mouse_position = event.globalPos()
                    return True
                elif event.button() == Qt.RightButton:
                    self.is_right_mouse_pressed = True
                    self.last_mouse_position = event.globalPos()
                    return True
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.RightButton:
                    self.is_right_mouse_pressed = False
                    return True
            elif event.type() == QEvent.MouseMove:
                if event.buttons() & Qt.LeftButton:
                    self.mouseMoveEvent(event)
                    return True
                elif self.is_right_mouse_pressed:
                    self.rightMouseMoveEvent(event)
                    return True
        return super().eventFilter(source, event)

    def refresh_black_canvas(self):
        canvas_size = 256 * self.grid_size  # Ensure the canvas covers the entire grid
        self.scene.setSceneRect(0, 0, canvas_size, canvas_size)
        black_rect = self.scene.addRect(
            QRectF(0, 0, canvas_size, canvas_size),
            brush=QBrush(QColor(0, 0, 0))
        )

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
            try:
                self.thread.progress_image.disconnect(self.update_image)
            except TypeError:
                # The signal was already disconnected
                pass

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
        self.refresh_black_canvas()

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
        # 计算视图中心点在场景中的位置
        view_center = self.graphics_view.viewport().rect().center()
        scene_center = self.graphics_view.mapToScene(view_center)
        
        # 更新中心位置
        self.center_position = (int(scene_center.x()), int(scene_center.y()))
        if self.thread:
            self.thread.set_center_position(self.center_position)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.thread and self.thread.isRunning():
            delta = event.globalPos() - self.last_mouse_position
            self.last_mouse_position = event.globalPos()
            
            # 更新滚动条位置
            self.graphics_view.horizontalScrollBar().setValue(
                self.graphics_view.horizontalScrollBar().value() - delta.x()
            )
            self.graphics_view.verticalScrollBar().setValue(
                self.graphics_view.verticalScrollBar().value() - delta.y()
            )
            
            # 更新中心位置
            self.update_center_position()

    def rightMouseMoveEvent(self, event):
        if self.thread and self.thread.isRunning():
            delta = event.globalPos() - self.last_mouse_position
            self.last_mouse_position = event.globalPos()
            
            # 根据水平移动距离计算缩放因子
            scale_change = 1 + (delta.x() / 100)  # 可以调整这个值来改变缩放速度
            self.scale_factor *= scale_change
            self.scale_factor = max(0.1, min(self.scale_factor, 5))
            
            # 应用缩放
            self.graphics_view.resetTransform()
            self.graphics_view.scale(self.scale_factor, self.scale_factor)
            
            # 更新中心位置
            self.update_center_position()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_position = event.globalPos()
        elif event.button() == Qt.RightButton:
            self.is_right_mouse_pressed = True
            self.last_mouse_position = event.globalPos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_position = QPoint()
        elif event.button() == Qt.RightButton:
            self.is_right_mouse_pressed = False

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