import sys
import os
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QProgressDialog,QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QComboBox, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton,QFileDialog,QGraphicsRectItem
)
from PyQt5.QtGui import QPixmap, QImage, QBrush, QColor,QPainter,QPen # Import QBrush and QColor
from PyQt5.QtCore import  Qt, QPoint, QRectF, QEvent,QTimer, QPropertyAnimation, QEasingCurve, QThread,QMetaObject # 添加 QEvent 导入
import torch
from decompress_backend import ImageLoaderThread, ImageSaverWorker  # Importing from backend
import subprocess
from decompress_images import load_model  # Assuming decompress_images is part of backend dependencies


class SatelliteImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.net = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.grid_size = 19
        self.bin_path = "classified_BIN"  # Directory containing bin files
        self.checkpoint = "save/0.05checkpoint_best.pth.tar"  # Model checkpoint file
        self.N = 64
        self.total_size = 0
        self.loading_blocks = {}
        self.size_label = None
        self.last_mouse_position = QPoint()
        self.is_right_mouse_pressed = False
        self.scale_factor = 1.0
        self.thread = None  # Reference to the current thread
        self.center_position = (0, 0)
        self.initUI()
        self.graphics_view.horizontalScrollBar().valueChanged.connect(self.update_thumbnail_viewport)
        self.graphics_view.verticalScrollBar().valueChanged.connect(self.update_thumbnail_viewport)
        self.graphics_view.viewport().installEventFilter(self)
        self.update_scale_label()
        self.thumbnail_update_timer = QTimer(self)
        self.thumbnail_update_timer.timeout.connect(self.update_thumbnail_while_loading)
      

    def initUI(self):
        layout = QVBoxLayout()

        # Create prefix dropdown and label
        input_layout = QHBoxLayout()
        self.load_label = QLabel("请选择卫星图")
        self.prefix_combo = QComboBox()
        self.prefix_combo.addItem("None")  # Add "None" as the default option
        self.load_prefixes()
       # 使用 lambda 忽略传递的 index 参数
        self.prefix_combo.currentIndexChanged.connect(lambda _: self.load_image())
        input_layout.addWidget(self.load_label)
        input_layout.addWidget(self.prefix_combo)

        # 修改后的按钮
        self.select_image_button = QPushButton("选择多块卫星图")
        self.select_image_button.clicked.connect(self.select_satellite_image)

        self.download_image_button = QPushButton("保存完整图片")
        self.download_image_button.setEnabled(False)  # 初始状态为不可点击
        self.download_image_button.clicked.connect(self.download_full_image)

        input_layout.addWidget(self.select_image_button)
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
        # 将视图中心设置到场景中心
        self.graphics_view.centerOn(self.center_position[0], self.center_position[1])
        
    
        self.setLayout(layout)
        self.setGeometry(100, 100, 1024, 768)
        self.setWindowTitle('Satellite Image Viewer')
        self.show()

        self.init_thumbnail(layout,input_layout)

        # 创建比例尺标签
        self.scale_label = QLabel()
        self.scale_label.setFixedWidth(50)
        self.scale_label.setAlignment(Qt.AlignCenter)

        # 创建总大小标签
        self.size_label = QLabel()
        self.size_label.setStyleSheet("""
            background-color: rgba(255, 255, 255, 180);
            padding: 5px;
            font-size: 16px;
            font-weight: bold;
        """)
        self.size_label.setAlignment(Qt.AlignCenter)

        # 创建左侧标签容器
        left_labels_container = QWidget()
        left_labels_layout = QVBoxLayout(left_labels_container)
        left_labels_layout.addWidget(self.size_label)
        left_labels_layout.addStretch(1)  # 添加弹性空间
        left_labels_layout.addWidget(self.scale_label)
        left_labels_layout.addStretch(1)  # 添加弹性空间
        left_labels_container.setFixedWidth(60)  # 设置固定宽度

        # 创建主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_labels_container)
        main_layout.addWidget(self.graphics_view)
        layout.addLayout(main_layout)

        # 确保总大小标签始终在最上层
        self.size_label.raise_()

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

    def select_satellite_image(self):
        # 打开文件对话框，让用户选择一个或多个 bin 文件
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择一块卫星图",
            self.bin_path,
            "Bin Files (*.bin);;All Files (*)",
            options=options
        )
        if not files:
            return  # 用户取消了选择

        # 解析选中的 bin 文件，提取前缀和位置信息
        bin_files = []
        for file_path in files:
            prefix, i, j = self.parse_bin_filename(os.path.basename(file_path))
            if prefix and i is not None and j is not None:
                bin_files.append((i, j, file_path))
            else:
                print(f"无法解析文件名: {file_path}")

        if not bin_files:
            print("未选择有效的 bin 文件。")
            return

        # 清空当前前缀选择
        self.prefix_combo.setCurrentIndex(0)  # 设置为 "None"

        # 加载选中的 bin 文件
        self.load_image(bin_files=bin_files)

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
        canvas_size = 256 * self.grid_size
        self.scene.setSceneRect(0, 0, canvas_size, canvas_size)
        # 更新未加载块的显示
        self.update_loading_blocks()

    def load_prefixes(self):
        prefixes = set()
        # 遍历压缩文件夹及其子文件夹
        for root, dirs, files in os.walk(self.bin_path):
            for filename in files:
                if filename.endswith('.bin'):
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        prefix = '_'.join(parts[:-2])
                        prefixes.add(prefix)
        self.prefix_combo.addItems(sorted(prefixes))

    def load_image(self, bin_files=None):
        """
        加载图像。可以通过传递 bin_files 参数来指定要加载的 bin 文件列表。
        如果 bin_files 为 None，则根据当前选择的前缀加载所有相关的 bin 文件。
        """
        if bin_files is None:
            # 从前缀组合框获取前缀
            prefix = self.prefix_combo.currentText()
            if prefix == "None":
                return  # 如果选择为 "None"，则不加载任何图像

            # 收集具有指定前缀的 bin 文件
            bin_files = []
            for root, dirs, files in os.walk(self.bin_path):
                for filename in files:
                    if filename.endswith('.bin'):
                        parsed = self.parse_bin_filename(filename)
                        if parsed:
                            file_prefix, i, j = parsed
                            if file_prefix == prefix:
                                bin_files.append((i, j, os.path.join(root, filename)))

            if not bin_files:
                print(f"No bin files found with prefix {prefix} in {self.bin_path}")
                return

        else:
            # 如果 bin_files 已经被提供（来自选择的文件），则无需过滤前缀
            pass

        # 计算总大小
        self.total_size = sum(os.path.getsize(file) for _, _, file in bin_files)
        self.update_size_label()  # 直接调用更新方法，无需使用 QMetaObject.invokeMethod


        # 初始化加载块信息
        self.loading_blocks = {(i, j): os.path.getsize(file) for i, j, file in bin_files}


        # 停止现有的线程（如果有）
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            try:
                self.thread.progress_image.disconnect(self.update_image)
            except TypeError:
                # 信号已经断开
                pass

        # 设置缩略图为灰色
        self.set_gray_thumbnail()

        if self.net is None:
            self.net = load_model(self.checkpoint, self.device, self.N)

        # 排序 bin 文件
        bin_files.sort()

        # 清除现有场景
        self.scene.clear()
        self.refresh_black_canvas()

        self.download_image_button.setEnabled(False)
        self.image_loaded = False

        # 启动图像加载线程
        self.thread = ImageLoaderThread(self.net, bin_files, self.bin_path, self.grid_size, self.device)
        self.thread.progress_image.connect(self.update_image)
        self.thread.finished.connect(self.on_image_load_complete)  # 连接完成信号
        self.thread.set_center_position(self.center_position)
        self.thread.start()

        # 启动缩略图更新定时器
        self.thumbnail_update_timer.start(1000)  # 每1000毫秒更新一次

    def parse_bin_filename(self, filename):
        """
        解析 bin 文件名，提取前缀和位置信息。

        返回:
            tuple: (prefix, i, j) 如果解析成功
            None: 如果解析失败
        """
        if not filename.endswith('.bin'):
            return None

        parts = filename.split('_')
        if len(parts) < 3:
            return None

        try:
            i = int(parts[-2])
            j = int(parts[-1].replace(".bin", ""))
            prefix = '_'.join(parts[:-2])
            return prefix, i, j
        except ValueError:
            return None


    def on_image_load_complete(self):
        self.download_image_button.setEnabled(True)
        self.image_loaded = True
        self.update_thumbnail()
        self.thumbnail_update_timer.stop()  # Stop the timer when loading is complete

    def update_thumbnail_while_loading(self):
        if self.thread and self.thread.isRunning():
            self.update_thumbnail()
        else:
            self.thumbnail_update_timer.stop()

    def update_size_label(self):
        if self.size_label is not None:
            total_size_mb = self.total_size / (1024 * 1024)
            vertical_text = '\n'.join(f"总大小: {total_size_mb:.2f} MB")
            self.size_label.setText(vertical_text)
            self.size_label.adjustSize()
            self.size_label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 在窗口大小变化时，确保 size_label 始终在左上角
        if self.size_label:
            self.size_label.move(10, 10)
            self.size_label.raise_()

    def update_image(self, pixmap, position):
        x, y = position
        item = QGraphicsPixmapItem(pixmap)
        item.setPos(x * 256, y * 256)
        self.scene.addItem(item)

        # 移除已加载块的信息
        if (x, y) in self.loading_blocks:
            del self.loading_blocks[x, y]


    def update_loading_blocks(self):
        for (x, y), size in self.loading_blocks.items():
            size_kb = size / 1024
            text_item = self.scene.addText(f"加载中\n{size_kb:.2f} KB")
            text_item.setPos(x * 256, y * 256)
            text_item.setDefaultTextColor(Qt.white)
            
            # 设置更大的字体
            font = text_item.font()
            font.setPointSize(14)  # 增大字体大小
            text_item.setFont(font)
            
            # 添加半透明背景
            rect = text_item.boundingRect()
            background = QGraphicsRectItem(rect)
            background.setBrush(QBrush(QColor(0, 0, 0, 128)))
            background.setParentItem(text_item)
            background.setZValue(-1)  # 确保背景在文本后面


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