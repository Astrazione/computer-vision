import sys
import warnings

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from UI.photoshopUI import Ui_MainWindow
from Logic.color_channel_transform import *
from Logic.blur_saturation import *
from Logic.visualization import *


class MainWindow(QMainWindow, Ui_MainWindow):
    source_image = None
    image = None
    image_copy = None

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
        self.color_channel_switch.clicked.connect(self.color_channel_switch_event)
        self.radio_red_channel.clicked.connect(self.show_image)
        self.radio_green_channel.clicked.connect(self.show_image)
        self.radio_blue_channel.clicked.connect(self.show_image)
        self.radio_grey_channel.clicked.connect(self.show_image)
        self.pushButton_vertical_reflection.clicked.connect(self.reflect_image_vertical)
        self.pushButton_horizontal_reflection.clicked.connect(self.reflect_image_horizontal)
        self.pushButton_swap_color_red_green.clicked.connect(self.swap_red_green_channels)
        self.pushButton_swap_color_green_blue.clicked.connect(self.swap_green_blue_channels)
        self.pushButton_swap_color_red_blue.clicked.connect(self.swap_red_blue_channels)
        self.pushButton_negative_color_red.clicked.connect(self.negate_red_channel)
        self.pushButton_negative_color_green.clicked.connect(self.negate_green_channel)
        self.pushButton_negative_color_blue.clicked.connect(self.negate_blue_channel)
        self.action_blur_4_pixels.triggered.connect(self.blur_4_pixels)
        self.action_blur_8_pixels.triggered.connect(self.blur_8_pixels)
        self.pushButton_calculate_brightness_profile.clicked.connect(self.show_brightness_profile)
        self.horizontalSlider_brightness.sliderMoved.connect(self.change_brightness)
        self.horizontalSlider_brightness_red.sliderMoved.connect(self.change_brightness_red)
        self.horizontalSlider_brightness_green.sliderMoved.connect(self.change_brightness_green)
        self.horizontalSlider_brightness_blue.sliderMoved.connect(self.change_brightness_blue)
        self.horizontalSlider_contrast.sliderReleased.connect(self.change_contrast)
        self.horizontalSlider_saturation.sliderMoved.connect(self.change_saturation)
        self.action_contrast_map_4_pixels.triggered.connect(self.show_contrast_map_4_pixels)
        self.action_contrast_map_8_pixels.triggered.connect(self.show_contrast_map_8_pixels)
        self.action_contrast_map_variable_pixels.triggered.connect(self.show_contrast_map_variable_pixels)
        self.pushButton_reload_hists.clicked.connect(self.reload_hists_event)
        self.actionReaload_hists.triggered.connect(self.reload_hists_event)
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

    def color_channel_switch_event(self):
        self.show_image()

    def show_image(self, need_to_rebuild: bool = True):
        if need_to_rebuild:
            self.build_image()

        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_item.setImage(image)
            self.create_brightness_hists()

    def build_image(self):
        self.image = self.source_image
        if self.color_channel_switch.isChecked():
            self.image = self.change_color_channel(self.image)
        # self.create_brightness_hists()

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
            self.label_pos_color.setText("Pos & color")
            self.label_color_intensity.setText("Color intensity")
            self.label_mean_std.setText("Mean & std")
            self.simple_show_image(self.image)
            return

        pos = event.pos()
        x, y = int(pos.x()), int(pos.y())

        x = int(np.clip(x, 0, self.image.shape[0] - 1))
        y = int(np.clip(y, 0, self.image.shape[1] - 1))

        b, g, r = self.image[x, y]

        self.label_pos_color.setText(f"Pixel position: ({x}, {y})  Value: {r}, {g}, {b}")
        self.label_color_intensity.setText(f"Intensity: {(r + g + b) // 3}")

        pixel_border_image, mean, std = get_image_with_calculated_pixel_border(self.image.copy(), self.border_size, x, y)

        self.label_mean_std.setText(f"Mean: {round(mean, 2)}  Std: {round(std, 2)}")

        self.simple_show_image(pixel_border_image)

    def simple_show_image(self, image, reload_hists=False):
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_item.updateImage(image)
            if reload_hists:
                self.create_brightness_hists()

    def reload_hists_event(self):
        self.create_brightness_hists()

    def get_image_res(self):
        return self.image.shape[0], self.image.shape[1]

    def reflect_image_vertical(self):
        self.image = self.image[:, ::-1, :]
        self.source_image = self.source_image[:, ::-1, :]
        self.show_image(False)

    def reflect_image_horizontal(self):
        self.image = self.image[::-1, :, :]
        self.source_image = self.source_image[::-1, :, :]
        self.show_image(False)

    def swap_red_green_channels(self):
        swap_channels(self.source_image, 2, 1)
        self.show_image()

    def swap_green_blue_channels(self):
        swap_channels(self.source_image, 1, 0)
        self.show_image()

    def swap_red_blue_channels(self):
        swap_channels(self.source_image, 2, 0)
        self.show_image()

    def negate_red_channel(self):
        negative_image_red_channel(self.source_image)
        self.show_image()

    def negate_green_channel(self):
        negative_image_green_channel(self.source_image)
        self.show_image()

    def negate_blue_channel(self):
        negative_image_blue_channel(self.source_image)
        self.show_image()

    def blur_4_pixels(self):
        self.source_image = blur_image_4_connectivity(self.source_image)
        self.show_image()

    def blur_8_pixels(self):
        self.source_image = blur_image_8_connectivity(self.source_image)
        self.show_image()

    def show_brightness_profile(self):
        str_number = self.spinBox_brightness_str_number.value()
        brightness_profile(self.image, str_number)

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

    def change_brightness(self):
        self.image = increase_brightness(self.source_image, self.horizontalSlider_brightness.value())
        self.simple_show_image(self.image)

    def change_brightness_red(self):
        self.image = increase_brightness_red_channel(self.source_image.copy(), self.horizontalSlider_brightness_red.value())
        self.simple_show_image(self.image)

    def change_brightness_green(self):
        self.image = increase_brightness_green_channel(self.source_image.copy(), self.horizontalSlider_brightness_green.value())
        self.simple_show_image(self.image)

    def change_brightness_blue(self):
        self.image = increase_brightness_blue_channel(self.source_image.copy(), self.horizontalSlider_brightness_blue.value())
        self.simple_show_image(self.image)

    def change_contrast(self):
        val = self.horizontalSlider_contrast.value()
        print(val)
        self.image = increase_contrast(self.source_image.copy(), self.horizontalSlider_contrast.value())
        self.simple_show_image(self.image)

    def change_saturation(self):
        self.image = increase_saturation(self.source_image.copy(), self.horizontalSlider_saturation.value())
        self.simple_show_image(self.image)

    def show_contrast_map_4_pixels(self):
        self.image = create_contrast_map(self.source_image.copy(), 4)
        self.simple_show_image(self.image)

    def show_contrast_map_8_pixels(self):
        self.image = create_contrast_map(self.source_image.copy(), 4)
        self.simple_show_image(self.image)

    def show_contrast_map_variable_pixels(self):
        self.image = create_contrast_map(self.source_image.copy(), self.spinBox_contrast_map.value())
        self.simple_show_image(self.image)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # !!!
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
