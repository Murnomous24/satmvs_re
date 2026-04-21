import numpy as np
import cv2
import math
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import random

def random_color(image):
    random_factor = np.random.randint(1, 301) / 100.
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # Image Color
    random_factor = np.random.randint(10, 201) / 100.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # Image Brightness
    random_factor = np.random.randint(10, 201) / 100.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # Image Contrast
    random_factor = np.random.randint(0, 301) / 100.
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # Image Sharpness

    return sharpness_image

def image_augment(image):
    image = random_color(image)

    return image

def dwt_aux_channels(gray_array, wavelet = "haar"):
    """
    Build two DWT edge-energy channels from a grayscale image.

    Input:
        gray_array: [H, W], float32, nominal range [0, 255]
    Output:
        ch2, ch3: [H, W], float32, non-negative
    """
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "dwt aux channel requires PyWavelets. Install it first, e.g. `uv pip install PyWavelets`."
        ) from exc

    coeffs = pywt.dwt2(gray_array, wavelet)
    _, (HL, LH, HH) = coeffs

    ch2_energy = np.sqrt(HL ** 2 + HH ** 2)
    ch3_energy = np.sqrt(LH ** 2 + HH ** 2)

    height, width = gray_array.shape
    ch2 = cv2.resize(ch2_energy, (width, height), interpolation = cv2.INTER_LINEAR)
    ch3 = cv2.resize(ch3_energy, (width, height), interpolation = cv2.INTER_LINEAR)

    return ch2.astype(np.float32), ch3.astype(np.float32)

def center_image(img): # TODO: why
    # scale 0~255 to 0~1

    img_array = np.array(img)
    img = img_array.astype(np.float32)

    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)
