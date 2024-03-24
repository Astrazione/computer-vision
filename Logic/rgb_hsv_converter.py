import numpy as np


# def rgb_to_hsv(rgb: np.ndarray):
#     normalized_rgb = rgb / 255.0
#     r, g, b = normalized_rgb[0], normalized_rgb[1], normalized_rgb[2]
#     s: float
#     v = normalized_rgb.max()
#     c_min = normalized_rgb.min()
#     delta = v - c_min
#
#     if v == 0:
#         s = 0
#     else:
#         s = delta / v
#
#     if s == 0:
#         h = 0
#     else:
#         cr = (v - r) / delta
#         cg = (v - g) / delta
#         cb = (v - b) / delta
#         if r == v:
#             h = cb - cg
#         elif g == v:
#             h = 2 + cr - cb
#         elif b == v:
#             h = 4 + cg - cr
#
#         h *= 60
#
#         if h < 0:
#             h += 360
#
#     return np.array([h, s, v])
#
#
# def rgb_to_hsv_array(rgb_bitmap: np.ndarray):
#     height, width, _ = rgb_bitmap.shape
#     hsv_bitmap = np.zeros(shape=rgb_bitmap.shape)
#
#     for y in range(height):
#         for x in range(width):
#             hsv_bitmap[y, x] = rgb_to_hsv(rgb_bitmap[y, x])
#
#     return hsv_bitmap
#
#
# def hsv_to_rgb(hsv_bitmap: np.ndarray):
#     h, s, v = hsv_bitmap
#     hi: int
#
#     hi = (h // 60) % 6
#     f = h / 60 - (h / 60)
#     p = v * (1 - s)
#     q = v * (1 - f * s)
#     t = v * (1 - (1 - f) * s)
#
#     if hi == 0:
#         r = v
#         g = t
#         b = p
#     if hi == 1:
#         r = q
#         g = v
#         b = p
#     if hi == 2:
#         r = p
#         g = v
#         b = t
#     if hi == 3:
#         r = p
#         g = q
#         b = v
#     if hi == 4:
#         r = t
#         g = p
#         b = v
#     if hi == 5:
#         r = v
#         g = p
#         b = q
#
#     return np.array([round(r * 255), round(g * 255), round(b * 255)])
#
#
# def hsv_to_rgb_array(hsv_bitmap: np.ndarray):
#     height, width, _ = hsv_bitmap.shape
#     rgb_bitmap = np.zeros(shape=hsv_bitmap.shape, dtype=np.uint8)
#
#     for y in range(height):
#         for x in range(width):
#             rgb_bitmap[y, x] = hsv_to_rgb(hsv_bitmap[y, x])
#
#     return rgb_bitmap


def rgb_to_hsv_vectorized(rgb_bitmap):
    rgb_normalized = rgb_bitmap.astype('float') / 255.0

    r, g, b = rgb_normalized[..., 0], rgb_normalized[..., 1], rgb_normalized[..., 2]
    max_c = np.max(rgb_normalized, axis=-1)
    min_c = np.min(rgb_normalized, axis=-1)
    delta = max_c - min_c

    h = np.zeros_like(max_c)
    v = max_c

    s = np.where(v == 0, 0, delta / v)

    mask = delta != 0
    idx = (r == max_c) & mask
    h[idx] = (g[idx] - b[idx]) / delta[idx]
    idx = (g == max_c) & mask
    h[idx] = 2.0 + (b[idx] - r[idx]) / delta[idx]
    idx = (b == max_c) & mask
    h[idx] = 4.0 + (r[idx] - g[idx]) / delta[idx]
    h = (h * 60) % 360
    h = np.where(delta == 0, 0, h)

    h = np.clip(h, 0, 360)
    s = np.clip(s, 0, 1)
    v = np.clip(v, 0, 1)

    return np.stack((h, s, v), axis=-1)


def hsv_to_rgb_vectorized(hsv_bitmap):
    h, s, v = hsv_bitmap[..., 0], hsv_bitmap[..., 1], hsv_bitmap[..., 2]

    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    r, g, b = np.zeros_like(c), np.zeros_like(c), np.zeros_like(c)

    mask = (h < 60)
    r[mask], g[mask], b[mask] = c[mask], x[mask], 0

    mask = (h >= 60) & (h < 120)
    r[mask], g[mask], b[mask] = x[mask], c[mask], 0

    mask = (h >= 120) & (h < 180)
    r[mask], g[mask], b[mask] = 0, c[mask], x[mask]

    mask = (h >= 180) & (h < 240)
    r[mask], g[mask], b[mask] = 0, x[mask], c[mask]

    mask = (h >= 240) & (h < 300)
    r[mask], g[mask], b[mask] = x[mask], 0, c[mask]

    mask = (h >= 300) & (h < 360)
    r[mask], g[mask], b[mask] = c[mask], 0, x[mask]

    r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    rgb_bitmap = np.stack([r, g, b], axis=-1).astype(np.uint8)

    return rgb_bitmap
