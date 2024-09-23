import sys
import os
from PyQt5.QtWidgets import (
    QProgressDialog,QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QComboBox, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton,QFileDialog,QGraphicsRectItem
)
from PyQt5.QtGui import QPixmap, QImage, QBrush, QColor,QPainter,QPen # Import QBrush and QColor
from PyQt5.QtCore import  Qt, QPoint, QRectF, QEvent,QTimer, QPropertyAnimation, QEasingCurve, QThread # 添加 QEvent 导入
import torch
from decompress_backend import ImageLoaderThread, ImageSaverWorker  # Importing from backend

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
        self.graphics_view.horizontalScrollBar().valueChanged.connect(self.update_thumbnail_viewport)
        self.graphics_view.verticalScrollBar().valueChanged.connect(self.update_thumbnail_viewport)
        self.graphics_view.viewport().installEventFilter(self)
        self.update_scale_label()

    def initUI(self):
        layout = QVBoxLayout()

        # Create prefix dropdown and label
        input_layout = QHBoxLayout()
        self.load_label = QLabel("请选择卫星图")
        self.prefix_combo = QComboBox()
        self.prefix_combo.addItem("None")  # Add "None" as the default option
        self.load_prefixes()
        self.prefix_combo.currentIndexChanged.connect(self.load_image)
        input_layout.addWidget(self.load_label)
        input_layout.addWidget(self.prefix_combo)

        # 新增的按钮
        self.open_folder_button = QPushButton("打开源文件夹")
        self.open_folder_button.clicked.connect(self.open_source_folder)

        self.download_image_button = QPushButton("保存完整图片")
        self.download_image_button.setEnabled(False)  # 初始状态为不可点击
        self.download_image_button.clicked.connect(self.download_full_image)

        input_layout.addWidget(self.open_folder_button)
        input_layout.addWidget(self.download_image_button)

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
    
        self.setLayout(layout)
        self.setGeometry(100, 100, 1024, 768)
        self.setWindowTitle('Satellite Image Viewer')
        self.show()

        self.init_thumbnail(layout,input_layout)

        # 创建比例尺标签
        self.scale_label = QLabel()
        self.scale_label.setFixedWidth(50)
        self.scale_label.setAlignment(Qt.AlignCenter)

        # 创建布局
        graphics_layout = QHBoxLayout()
        graphics_layout.addWidget(self.scale_label)
        graphics_layout.addWidget(self.graphics_view)
        layout.addLayout(graphics_layout)

    def update_scale_label(self):
        scale_text = f"{self.scale_factor:.2f}x"
        vertical_text = '<br>'.join(scale_text)
        self.scale_label.setText(vertical_text)
        self.scale_label.setStyleSheet("font-size: 24px;")  # Double the font size
        self.scale_label.setFixedWidth(40)  # Adjust width as needed
        self.scale_label.setAlignment(Qt.AlignCenter)

    def init_thumbnail(self, layout, input_layout):
        layout.addLayout(input_layout)
        
        # 使用提示标签
        usage_tip = QLabel("使用说明：鼠标左键拖动移动图像，鼠标右键左右拖动进行缩放")
        usage_tip.setAlignment(Qt.AlignCenter)
        usage_tip.setStyleSheet("color: black; font-family: SimSun, serif; font-size: 14px;")
        layout.addWidget(usage_tip)

        # 创建一个包含主画布和缩略图的容器widget
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # 添加主画布到容器
        container_layout.addWidget(self.graphics_view)

        # 创建缩略图视图
        self.thumbnail_view = QGraphicsView()
        self.thumbnail_view.setFixedSize(200, 200)
        self.thumbnail_scene = QGraphicsScene()
        self.thumbnail_scene.setSceneRect(0, 0, 200, 200)
        self.thumbnail_view.setScene(self.thumbnail_scene)
        self.thumbnail_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbnail_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 初始化缩略图为灰色
        self.set_gray_thumbnail()

        # 设置缩略图的样式，使其看起来像悬浮
        self.thumbnail_view.setStyleSheet("""
            background: transparent;
            border: 2px solid lightgray;
            border-radius: 5px;
        """)

        # 创建一个新的布局来放置缩略图
        thumbnail_layout = QHBoxLayout()
        thumbnail_layout.addStretch(1)
        thumbnail_layout.addWidget(self.thumbnail_view)
        thumbnail_layout.setContentsMargins(0, 0, 10, 10)  # 右下角留出一些边距

        # 将缩略图布局叠加到容器上
        container_layout.addLayout(thumbnail_layout)

        # 将容器添加到主布局
        layout.addWidget(container)

        self.setLayout(layout)

        # 设置缩略图在主画布上的位置
        self.thumbnail_view.setParent(self.graphics_view)
        self.thumbnail_view.move(self.graphics_view.width() - 210, self.graphics_view.height() - 210)
        self.thumbnail_view.raise_()  # 确保缩略图在最上层

        # 连接窗口大小变化信号到更新缩略图位置的槽函数
        self.graphics_view.resizeEvent = self.update_thumbnail_position

    def set_gray_thumbnail(self):
        gray_pixmap = QPixmap(200, 200)
        gray_pixmap.fill(QColor(200, 200, 200))  # 设置为浅灰色
        self.thumbnail_scene.clear()
        self.thumbnail_scene.addPixmap(gray_pixmap)

    def update_thumbnail_position(self, event):
        # 更新缩略图位置
        self.thumbnail_view.move(self.graphics_view.width() - 210, self.graphics_view.height() - 210)
        # 调用原始的 resizeEvent
        super(QGraphicsView, self.graphics_view).resizeEvent(event)

    def update_thumbnail(self):
        # 清除之前的缩略图
        self.thumbnail_scene.clear()

        # 获取主场景的图像
        scene_rect = self.scene.sceneRect()
        image = QImage(scene_rect.size().toSize(), QImage.Format_ARGB32)
        painter = QPainter(image)
        self.scene.render(painter)
        painter.end()

        # 缩放图像以适应缩略图视图
        thumbnail_size = self.thumbnail_view.viewport().size()
        scaled_image = image.scaled(thumbnail_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap = QPixmap.fromImage(scaled_image)

        # 在缩略图场景中显示图像
        self.thumbnail_scene.addPixmap(pixmap)

        # 更新视口矩形
        self.update_thumbnail_viewport()

    def update_thumbnail_viewport(self):
        # Remove previous viewport rectangle
        for item in self.thumbnail_scene.items():
            if isinstance(item, QGraphicsRectItem) and item.data(0) == 'viewport_rect':
                self.thumbnail_scene.removeItem(item)
        
        # Get the main scene and thumbnail dimensions
        scene_rect = self.scene.sceneRect()
        thumbnail_rect = self.thumbnail_scene.sceneRect()
        
        # Calculate scaling ratios
        x_ratio = thumbnail_rect.width() / scene_rect.width()
        y_ratio = thumbnail_rect.height() / scene_rect.height()
        
        # Calculate visible area dimensions
        visible_width = self.graphics_view.viewport().width() / self.scale_factor
        visible_height = self.graphics_view.viewport().height() / self.scale_factor
        
        # Calculate rectangle position and size in thumbnail coordinates
        x = (self.center_position[0] - visible_width / 2) * x_ratio
        y = (self.center_position[1] - visible_height / 2) * y_ratio
        width = visible_width * x_ratio
        height = visible_height * y_ratio
        print("缩略图白色空心框位置：",x,y,width,height)
        # Create and add the viewport rectangle to the thumbnail scene
        rect_item = QGraphicsRectItem(x, y, width, height)
        rect_item.setPen(QPen(QColor(255, 255, 255)))
        rect_item.setData(0, 'viewport_rect')
        self.thumbnail_scene.addItem(rect_item)

    def open_source_folder(self):
        # 打开存储 bin 文件的文件夹
        path = os.path.abspath(self.bin_path)
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])

    def download_full_image(self):
        if self.image_loaded:
            # 让用户选择保存位置和文件名
            save_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG Files (*.png);;All Files (*)")
            if not save_path:
                return  # 用户取消了保存操作

            # 创建并显示进度对话框
            progress_dialog = QProgressDialog("正在保存图片...", "取消", 0, 0, self)
            progress_dialog.setWindowTitle("保存进度")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setCancelButton(None)  # 移除取消按钮
            progress_dialog.setRange(0, 0)  # 设置为不确定的进度条

            # 创建保存线程和工作器
            self.save_thread = QThread()
            self.saver = ImageSaverWorker(self.scene, save_path)
            self.saver.moveToThread(self.save_thread)

            # 连接信号和槽
            self.save_thread.started.connect(self.saver.save_image)
            self.saver.finished.connect(self.save_thread.quit)
            self.saver.finished.connect(self.saver.deleteLater)
            self.save_thread.finished.connect(self.save_thread.deleteLater)
            self.save_thread.finished.connect(progress_dialog.close)
            self.saver.success.connect(lambda: self.show_popup_message("图片保存成功"))
            self.saver.error.connect(self.show_popup_message)

            # 启动线程
            self.save_thread.start()

            # 显示进度对话框
            progress_dialog.exec_()

    def cancel_save(self):
        if self.saver:
            self.saver.cancel()
        if self.save_thread:
            self.save_thread.quit()
            self.save_thread.wait()  

    def show_popup_message(self, message):
        popup = QLabel(message, self)
        popup.setStyleSheet("""
            background-color: rgba(0, 0, 0, 180);
            color: white;
            padding: 10px;
            border-radius: 5px;
        """)
        popup.setAlignment(Qt.AlignCenter)
        popup.adjustSize()

        # 将弹窗居中显示
        popup.move(
            self.width() // 2 - popup.width() // 2,
            self.height() // 2 - popup.height() // 2
        )

        # 创建淡出动画
        fade_out = QPropertyAnimation(popup, b"windowOpacity")
        fade_out.setDuration(1000)  # 1秒
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.OutQuad)

        # 显示弹窗
        popup.show()

        # 使用 QTimer 来控制整个过程
        def start_fade_out():
            fade_out.start()

        def remove_popup():
            popup.deleteLater()
            # 确保动画对象也被删除
            fade_out.deleteLater()

        # 1秒后开始淡出动画
        QTimer.singleShot(1000, start_fade_out)

        # 2秒后（显示1秒 + 淡出1秒）删除弹窗
        QTimer.singleShot(2000, remove_popup)

        # 保持对动画的引用，防止被过早回收
        popup.fade_out_animation = fade_out


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
        prefix = self.prefix_combo.currentText()
        if prefix == "None":
            return  # Do nothing if "None" is selected
        # Stop existing thread if running
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            try:
                self.thread.progress_image.disconnect(self.update_image)
            except TypeError:
                # The signal was already disconnected
                pass
        # 设置缩略图为灰色
        self.set_gray_thumbnail()

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

        self.download_image_button.setEnabled(False)
        self.image_loaded = False
        # Start the image loading thread
        self.thread = ImageLoaderThread(self.net, bin_files, self.bin_path, self.grid_size, self.device)
        self.thread.progress_image.connect(self.update_image)
        self.thread.finished.connect(self.on_image_load_complete)  # Connect the finished signal here
        self.thread.set_center_position(self.center_position)
        self.thread.start()

    def on_image_load_complete(self):
        self.download_image_button.setEnabled(True)
        self.image_loaded = True
        self.update_thumbnail()

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
        if event.buttons() & Qt.LeftButton:
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
        if event.buttons() & Qt.RightButton:
            delta = event.globalPos() - self.last_mouse_position
            self.last_mouse_position = event.globalPos()
            
            # 根据水平移动距离计算缩放因子
            scale_change = 1 + (delta.x() / 100)
            self.scale_factor *= scale_change
            self.scale_factor = max(0.1, min(self.scale_factor, 5))
            
            # 应用缩放
            self.graphics_view.resetTransform()
            self.graphics_view.scale(self.scale_factor, self.scale_factor)
            
            # 更新中心位置和比例尺标签
            self.update_center_position()
            self.update_scale_label()

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