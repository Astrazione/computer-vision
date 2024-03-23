import math
import numpy as np
import matplotlib.pyplot as plt


# neighbor_count - min = 4, а также из этого числа можно взять корень прибавив 1
def create_contrast_map(image, neighbor_count=4):
    neighbor_indices = []

    if neighbor_count == 4:
        neighbor_indices = [(0, 1), (1, 0), (0, 0), (0, -1), (-1, 0)]
    elif int(math.sqrt(neighbor_count + 1)) != math.sqrt(neighbor_count + 1):
        return image
    else:
        n = int(math.sqrt(neighbor_count))
        neighbor_indices = [(i, j) for i in range(-n // 2, n // 2 + 1) for j in range(-n // 2, n // 2 + 1)]

    height, width, _ = image.shape

    contrast_map = np.zeros_like(image, dtype=np.uint8)

    avg_r, avg_g, avg_b = image[:, :, 2].mean(), image[:, :, 1].mean(), image[:, :, 0].mean()
    avg = (float(avg_r) + float(avg_g) + float(avg_b)) // 3

    for i in range(height - 1):
        for j in range(width - 1):
            neighbor_min = avg
            neighbor_max = avg
            for index in neighbor_indices:
                neighborPixel  = image[min(max(0, i + index[0]), height - 1), min(max(0, j + index[1]), width - 1)]
                tmp = (int(neighborPixel[0]) + int(neighborPixel[1]) + int(neighborPixel[2])) // 3

                if tmp < neighbor_min:
                    neighbor_min = tmp
                elif tmp > neighbor_max:
                    neighbor_max = tmp

            contrast_coef = round((neighbor_max - neighbor_min) / (neighbor_max + neighbor_min) * 255)
            contrast_map[i, j] = [contrast_coef, contrast_coef, contrast_coef]


    return contrast_map


def get_image_with_calculated_pixel_border(image, border_size, x, y):
    width, height = image.shape[0], image.shape[1]
    image_with_pixel_border = image.copy()

    mean: float = 0
    std = 0
    delta = (border_size + 1) / 2
    x_start, x_end = int(max(x - delta, 0)), int(min(x + delta + 1, width))
    y_start, y_end = int(max(y - delta, 0)), int(min(y + delta + 1, height))

    pixel_count = (x_end - x_start - 2) * (y_end - y_start - 2)

    for x_index in range(x_start, x_end):
        for y_index in range(y_start, y_end):
            if not (x_index == x - delta or x_index == x + delta or y_index == y - delta or y_index == y + delta):
                pixel = image[x_index, y_index]
                mean = mean + (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3

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


def brightness_profile(image, index_row):
    row = image[index_row]

    brightness_values = [np.mean(pixel) for pixel in row]

    fig, ax = plt.subplots()
    ax.plot(brightness_values)
    ax.set_title("Brightness Profile")
    ax.set_xlabel("Pixel Index")
    ax.set_ylabel("Brightness")
    ax.set_ylim(0, 255)

    fig.show()
