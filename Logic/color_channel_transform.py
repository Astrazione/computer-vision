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


def negative_image(image):
    height, width, _ = image.shape

    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            new_pixel = tuple(255-p for p in pixel)
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
