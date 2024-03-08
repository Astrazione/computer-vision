import sys
import warnings

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from UI.photoshopUI import Ui_MainWindow
from Logic.color_channel_transform import *


class MainWindow(QMainWindow, Ui_MainWindow):
    source_image = None
    image = None

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.action_open.triggered.connect(self.open_image)
        self.action_save.triggered.connect(self.save_image)
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_item = pg.ImageItem()
        self.image_view.addItem(self.image_item)
        self.image_item.hoverEvent = self.image_hover_event
        self.red_plot: pg.PlotItem = self.graphics_widget.addPlot(title="red")
        self.green_plot: pg.PlotItem = self.graphics_widget.addPlot(title="green")
        self.blue_plot: pg.PlotItem = self.graphics_widget.addPlot(title="blue")
        self.color_channel_switch.clicked.connect(self.show_image)
        self.radio_red_channel.clicked.connect(self.show_image)
        self.radio_green_channel.clicked.connect(self.show_image)
        self.radio_blue_channel.clicked.connect(self.show_image)
        self.radio_grey_channel.clicked.connect(self.show_image)

        self.border_size = 11  # px

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if not file_name:
            return

        self.source_image = cv2.imread(file_name).transpose(1, 0, 2)
        self.show_image()

    def show_image(self):
        self.build_image()

        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_item.setImage(image)

    def build_image(self):
        self.image = self.source_image
        self.image = self.change_color_channel(self.image)
        self.create_brightness_hists()

    def change_color_channel(self, image):
        if not self.color_channel_switch.isChecked():
            return image
        elif self.radio_red_channel.isChecked():
            return extract_red_channel(image)
        elif self.radio_green_channel.isChecked():
            return extract_green_channel(image)
        elif self.radio_blue_channel.isChecked():
            return extract_blue_channel(image)
        elif self.radio_grey_channel.isChecked():
            return convert_to_grayscale(image)
        else:
            return image

    def save_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)

        cv2.imwrite(file_name, self.image.transpose(1, 0, 2))

    def image_hover_event(self, event):
        if event.isExit():
            self.label_pos_color.setText("")
            self.label_color_intensity.setText("")
            self.simple_show_image(self.image)
            return

        pos = event.pos()
        x, y = int(pos.x()), int(pos.y())

        x = int(np.clip(x, 0, self.image.shape[0] - 1))
        y = int(np.clip(y, 0, self.image.shape[1] - 1))

        b, g, r = self.image[x, y]

        self.label_pos_color.setText(f"pixel: ({x}, {y})  value: {r}, {g}, {b}")
        self.label_color_intensity.setText(f"Интенсивность: {(r + g + b) // 3}")
        self.simple_show_image(self.get_image_with_pixel_border(self.image, self.border_size, x, y))

    def simple_show_image(self, image):
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_item.setImage(image)

    @staticmethod
    def get_image_with_pixel_border(image, border_size, x, y):
        width = image.shape[0]
        height = image.shape[1]

        image_with_pixel_border = image.copy()

        delta = (border_size - 1) / 2
        x_start, x_end = int(max(x - delta, 0)), int(min(x + delta + 1, width))
        y_start, y_end = int(max(y - delta, 0)), int(min(y + delta + 1, height))

        for x_index in range(x_start, x_end):
            for y_index in range(y_start, y_end):
                if x_index == x_start or x_index == x_end - 1 or y_index == y_start or y_index == y_end - 1:
                    image_with_pixel_border[x_index, y_index] = np.array([0, 255, 255])

        return image_with_pixel_border

    def create_brightness_hists(self):
        width = self.image.shape[0]
        height = self.image.shape[1]

        r_channel, g_channel, b_channel = np.zeros(256), np.zeros(256), np.zeros(256)

        for x in range(width):
            for y in range(height):
                bgr = self.image[x, y]
                b_channel[bgr[0]] += 1
                g_channel[bgr[1]] += 1
                r_channel[bgr[2]] += 1

        pixels_count = width * height
        b_channel /= pixels_count
        g_channel /= pixels_count
        r_channel /= pixels_count

        self.show_red_hist(r_channel)
        self.show_green_hist(g_channel)
        self.show_blue_hist(b_channel)

    def show_red_hist(self, r_channel):
        self.red_plot.clearPlots()
        self.red_plot.plot(x=list(range(256)), y=r_channel)

    def show_green_hist(self, g_channel):
        self.green_plot.clearPlots()
        self.green_plot.plot(x=list(range(256)), y=g_channel)

    def show_blue_hist(self, b_channel):
        self.blue_plot.clearPlots()
        self.blue_plot.plot(x=list(range(256)), y=b_channel)





if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # !!!
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
