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

def center_image(img): # TODO: why
    # scale 0~255 to 0~1

    img_array = np.array(img)
    img = img_array.astype(np.float32)

    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)