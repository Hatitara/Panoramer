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


def apply_sobel(image: np.ndarray, direction: str) -> np.ndarray:
    '''
    Applies a custom Sobel filter to detect edges in a specific direction.

    :param image: Grayscale input image.
    :param direction: 'x' for horizontal gradients, 'y' for vertical gradients.
    :return: Image with Sobel filter applied.
    '''
    image = image.astype(np.float32)

    if direction == 'x':
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
    elif direction == 'y':
        kernel = np.array([[-1, -2, -1],
                           [0,  0,  0],
                           [1,  2,  1]], dtype=np.float32)
    else:
        raise ValueError("Direction must be 'x' or 'y'")

    h, w = image.shape
    output = np.zeros((h-2, w-2), dtype=np.float32)

    for i in range(h - 2):
        for j in range(w - 2):
            region = image[i:i+3, j:j+3]
            output[i, j] = np.sum(region * kernel)

    return output


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    '''
    Applies a custom Gaussian blur to an image.

    :param image: Grayscale input image.
    :param kernel_size: Size of the Gaussian kernel (must be odd).
    :param sigma: Standard deviation of the Gaussian function.
    :return: Blurred image.
    '''
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)

    h, w = image.shape
    output = np.zeros((h - kernel_size + 1, w - kernel_size + 1), dtype=np.float32)

    for i in range(h - kernel_size + 1):
        for j in range(w - kernel_size + 1):
            region = image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)

    return output
