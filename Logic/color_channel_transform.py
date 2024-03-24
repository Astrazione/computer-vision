import numpy as np
import cv2
from Logic.rgb_hsv_converter import *


def extract_blue_channel(image):
    blue_image = np.zeros_like(image)
    blue_image[:, :, 0] = image[:, :, 0]
    return blue_image


def extract_green_channel(image):
    green_image = np.zeros_like(image)
    green_image[:, :, 1] = image[:, :, 1]
    return green_image


def extract_red_channel(image):
    red_image = np.zeros_like(image)
    red_image[:, :, 2] = image[:, :, 2]
    return red_image


def convert_to_grayscale(image):
    height, width, _ = image.shape
    # gray_image = np.zeros((height, width), dtype=np.uint8)\
    gray_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            gray_value = sum(image[i, j]) // 3
            gray_image[i, j]: np.ndarray = np.array([gray_value, gray_value, gray_value])
    return gray_image


def negative_image(image):
    height, width, _ = image.shape

    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            new_pixel = np.array([255-p for p in pixel])
            image[y, x] = new_pixel
    return image


def negative_image_red_channel(image):
    negative_red_image = negative_image(extract_red_channel(image.copy()))
    image[:, :, 2] = negative_red_image[:, :, 2]


def negative_image_green_channel(image):
    negative_green_image = negative_image(extract_green_channel(image.copy()))
    image[:, :, 1] = negative_green_image[:, :, 1]


def negative_image_blue_channel(image):
    negative_blue_image = negative_image(extract_blue_channel(image.copy()))
    image[:, :, 0] = negative_blue_image[:, :, 0]


def swap_channels(image, channel1, channel2):
    temp_image = image.copy()
    image[:, :, channel1], image[:, :, channel2] = temp_image[:, :, channel2], temp_image[:, :, channel1]


def increase_brightness(image, value):
    image_copy = image.copy().astype(np.uint16)
    return np.clip(image_copy + value, 0, 255).astype(np.uint8)


def increase_brightness_red_channel(image, value):
    red_image = extract_red_channel(image.copy())
    red_image = increase_brightness(red_image, value)
    image[:, :, 2] = red_image[:, :, 2]
    return image


def increase_brightness_green_channel(image, value):
    green_image = extract_green_channel(image.copy())
    green_image = increase_brightness(green_image, value)
    image[:, :, 1] = green_image[:, :, 1]
    return image


def increase_brightness_blue_channel(image, value):
    blue_image = extract_blue_channel(image.copy())
    blue_image = increase_brightness(blue_image, value)
    image[:, :, 0] = blue_image[:, :, 0]
    return image


# def increase_contrast(image, factor):
#     height, width, channels = image.shape
#
#     factor = max(-255, min(255, factor))
#     #Рассчет коэффициента корректности
#     factor = (259 * (factor + 255)) / (255 * (259 - factor))
#
#     for y in range(height):
#         for x in range(width):
#             for c in range(channels):
#                 #128 - среднее значение пикселя
#                 new_pixel_value = min(255, max(0, factor * (image[y, x, c] - 128) + 128))
#                 image[y, x, c] = new_pixel_value
#
#     return image

# def increase_contrast(image, factor):
#     height, width, channels = image.shape
#
#     factor = (100.0 + factor) / 100.0
#     factor *= factor
#
#     for y in range(height):
#         for x in range(width):
#             image[y, x, 0] = np.clip(((image[y, x, 0] / 255 - 0.5) * factor + 0.5) * 255, 0, 255)
#             image[y, x, 1] = np.clip(((image[y, x, 1] / 255 - 0.5) * factor + 0.5) * 255, 0, 255)
#             image[y, x, 2] = np.clip(((image[y, x, 2] / 255 - 0.5) * factor + 0.5) * 255, 0, 255)
#
#     return image

def increase_contrast(image, factor):
    factor = (100.0 + factor) / 100.0

    normalized_image = image.astype(np.float64) / 255.0  # Normalize image to 0-1 range
    normalized_image -= 0.5
    adjusted_image = normalized_image * factor + 0.5
    clipped_image = np.clip(adjusted_image * 255.0, 0, 255).astype(np.uint8)  # Clip and convert back to uint8

    return clipped_image


def increase_saturation(image, value):
    hsv = rgb_to_hsv_vectorized(image)

    s = hsv[:, :, 1] + value / 255
    s = np.clip(s, 0, 1)
    hsv[:, :, 1] = s

    rgb = hsv_to_rgb_vectorized(hsv)
    return rgb

