import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage
from PyQt5.QtCore import Qt, QRect
import torch
from PIL import Image
import numpy as np
from decompress_images import load_model, decompress_bin_file, assemble_grid


class SatelliteImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.net = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.grid_size = 19
        self.bin_path = "compressedBIN"  # 指定存放bin文件的文件夹
        self.checkpoint = "save/0.05checkpoint_best.pth.tar"  # 指定模型checkpoint文件
        self.N = 64

    def initUI(self):
        layout = QVBoxLayout()

        # 创建输入框和按钮
        input_layout = QHBoxLayout()
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText("Enter prefix")
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        input_layout.addWidget(self.prefix_input)
        input_layout.addWidget(self.load_button)
        layout.addLayout(input_layout)

        # 创建滚动区域来显示图像
        self.scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)
        self.setGeometry(100, 100, 1024, 1024)
        self.setWindowTitle('Satellite Image Viewer')
        self.show()

    def load_image(self):
        prefix = self.prefix_input.text()
        if not prefix:
            return

        if self.net is None:
            self.net = load_model(self.checkpoint, self.device, self.N)

        # 收集bin文件
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

        # 排序bin文件
        bin_files.sort()

        images = []
        for i, j, bin_path in bin_files:
            json_filename = f"{os.path.splitext(os.path.basename(bin_path))[0]}.json"
            json_path = os.path.join(self.bin_path, json_filename)
            if not os.path.exists(json_path):
                print(f"JSON file {json_path} not found for bin file {bin_path}. Skipping.")
                continue
            x_hat = decompress_bin_file(self.net, bin_path, json_path, self.device)
            images.append({'position': (j, i), 'image': x_hat.squeeze(0)})


        # 组装网格
        grid_image = assemble_grid(images, self.grid_size)

        # 转换为QPixmap并显示
        grid_image_np = np.array(grid_image)
        height, width, channel = grid_image_np.shape
        bytes_per_line = 3 * width
        qimage = QImage(grid_image_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SatelliteImageViewer()
    sys.exit(app.exec_())