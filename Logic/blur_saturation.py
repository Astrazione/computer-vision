import numpy as np

def blur_image_4_connectivity(image):
    height, width, channels = image.shape
    blurred_image = np.zeros_like(image, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                total_pixel = 0
                count = 1

                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        total_pixel += image[ny, nx, c]
                        count += 1
                blurred_image[y, x, c] = total_pixel // count

    return blurred_image


def blur_image_8_connectivity(image):
    height, width, channels = image.shape
    blurred_image = np.zeros_like(image, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                total_pixel = 0
                count = 1

                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            total_pixel += image[ny, nx, c]
                            count += 1
                blurred_image[y, x, c] = total_pixel // count
    return blurred_image
