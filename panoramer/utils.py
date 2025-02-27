'''
Helper functions for general tasks.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image(image: cv2.typing.MatLike, width: int = None, height: int = None) -> cv2.typing.MatLike:
    '''
    Resizes the image to the specified width and height.
    '''
    if width is None and height is None:
        return image

    h, w = image.shape[:2]

    if width is None:
        scale = height / float(h)
        width = int(w * scale)
    else:
        scale = width / float(w)
        height = int(h * scale)

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def blend_images(image1: cv2.typing.MatLike, image2: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    Blends two images together to create a seamless transition.
    '''
    pass


def normalize_brightness_contrast(image: cv2.typing.MatLike, target_brightness: float = 128, target_contrast: float = 50) -> cv2.typing.MatLike:
    '''
    Normalizes brightness and contrast of an image to a target level.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, std_dev = np.mean(gray), np.std(gray)

    if std_dev == 0:
        std_dev = 1e-6

    normalized = (image - mean) * (target_contrast /
                                   std_dev) + target_brightness
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized


def plot_brightness_histogram(image: cv2.typing.MatLike, title: str = "Brightness Histogram") -> None:
    '''
    Plots the brightness histogram of a grayscale image.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    plt.figure(figsize=(6, 4))
    plt.plot(hist, color='black')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])
    plt.grid()
    plt.show()
