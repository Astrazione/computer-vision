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
        # self.brightness_hist_item = pg.HistogramLUTItem()
        # self.brightness_hist_widget.addItem(self.brightness_hist_item)
        # self.brightness_hist_item.setImageItem(self.image_item)
        self.brightness_hist.removeItem(self.brightness_hist.item)
        self.brightness_hist.addItem(pg.HistogramLUTItem(orientation='bottom'))

        self.color_channel_switch.clicked.connect(self.show_image)
        self.radio_red_channel.clicked.connect(self.show_image)
        self.radio_green_channel.clicked.connect(self.show_image)
        self.radio_blue_channel.clicked.connect(self.show_image)
        self.radio_grey_channel.clicked.connect(self.show_image)

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

    # def set_pixel_border(self):

    def image_hover_event(self, event):
        if event.isExit():
            self.label_pos_color.setText("")
            self.label_color_intensity.setText("")
            return

        pos = event.pos()
        x, y = int(pos.x()), int(pos.y())

        x = int(np.clip(x, 0, self.image.shape[0] - 1))
        y = int(np.clip(y, 0, self.image.shape[1] - 1))

        b, g, r = self.image[x, y]

        self.label_pos_color.setText(f"pixel: ({x}, {y})  value: {r}, {g}, {b}")
        self.label_color_intensity.setText(f"Интенсивность: {(r + g + b) // 3}")

    # def set_pixel_border(self, x, y, color):




if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # !!!
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())