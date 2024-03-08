import numpy as np


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
