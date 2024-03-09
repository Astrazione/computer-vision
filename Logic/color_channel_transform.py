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


def get_image_with_calculated_pixel_border(image, border_size, x, y):
    width, height = image.shape[0], image.shape[1]
    image_with_pixel_border = image.copy()
    pixel_count = width * height

    mean: int = 0
    std: int = 0
    delta = (border_size + 1) / 2
    x_start, x_end = int(max(x - delta, 0)), int(min(x + delta + 1, width))
    y_start, y_end = int(max(y - delta, 0)), int(min(y + delta + 1, height))

    for x_index in range(x_start, x_end):
        for y_index in range(y_start, y_end):
            if not (x_index == x - delta or x_index == x + delta or y_index == y - delta or y_index == y + delta):
                pixel = sum(image[x_index, y_index]) // 3
                mean += pixel

    mean /= pixel_count

    for x_index in range(x_start, x_end):
        for y_index in range(y_start, y_end):
            if x_index == x - delta or x_index == x + delta or y_index == y - delta or y_index == y + delta:
                image_with_pixel_border[x_index, y_index] = np.array([0, 255, 255])
            else:
                pixel = sum(image[x_index, y_index]) // 3
                std += (pixel - mean) ** 2

    std = np.sqrt(std / pixel_count)

    return image_with_pixel_border, mean, std
