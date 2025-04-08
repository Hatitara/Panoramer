'''
Helper functions for general tasks.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from typing import Tuple


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

def inverse_homography(H: np.ndarray) -> np.ndarray:
    '''
    Manually computes the inverse of a 3x3 homography matrix using the determinant and cofactors.

    :param H: A 3x3 homography matrix
    :return: The manually computed inverse of H
    '''
    h11, h12, h13 = H[0, :]
    h21, h22, h23 = H[1, :]
    h31, h32, h33 = H[2, :]

    det_H = (h11 * (h22 * h33 - h23 * h32)
           - h12 * (h21 * h33 - h23 * h31)
           + h13 * (h21 * h32 - h22 * h31))

    C = np.array([
        [ (h22 * h33 - h23 * h32), -(h12 * h33 - h13 * h32),  (h12 * h23 - h13 * h22)],
        [-(h21 * h33 - h23 * h31),  (h11 * h33 - h13 * h31), -(h11 * h23 - h13 * h21)],
        [ (h21 * h32 - h22 * h31), -(h11 * h32 - h12 * h31),  (h11 * h22 - h12 * h21)]
    ])

    adj_H = C.T
    H_inv = (1 / det_H) * adj_H

    return H_inv.T


def gram_schmidt_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Performs QR decomposition using the Gram-Schmidt process.

    :param A: Input matrix of shape (m, n).
    :return: Tuple (Q, R) where Q is an orthonormal matrix and R is upper triangular such that A = Q @ R.
    '''
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = norm(v)
        if R[j, j] < 1e-10:
            Q[:, j] = np.zeros(m)
        else:
            Q[:, j] = v / R[j, j]

    return Q, R


def dominant_eig(A: np.ndarray, max_iterations: int = 1000) -> Tuple[float, np.ndarray]:
    '''
    Finds the dominant eigenvalue and eigenvector using the power iteration method.

    :param A: Square matrix.
    :param max_iterations: Number of iterations to run the power method.
    :return: Tuple (eigenvalue, eigenvector) corresponding to the dominant eigenvalue.
    '''
    n = A.shape[1]

    b_k = np.random.rand(n)

    for _ in range(max_iterations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = norm(b_k1)

        if b_k1_norm < 1e-10:
            return 0.0, np.zeros(n)

        b_k = b_k1 / b_k1_norm

    # Rayleigh quotient
    eigenvalue = np.dot(np.conj(b_k).T, np.dot(A, b_k)) / \
        np.dot(np.conj(b_k).T, b_k)
    eigenvector = b_k

    return eigenvalue, eigenvector


def eig(A: np.ndarray, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Estimates all eigenvalues and eigenvectors of a symmetric matrix using repeated deflation and power iteration.

    :param A: Symmetric matrix (typically A^T @ A in SVD).
    :param max_iterations: Number of iterations for each power iteration.
    :return: Tuple (eigenvalues, eigenvectors) as arrays.
    '''
    eigenvalues = []
    eigenvectors = []
    A_c = A.copy()
    n = A_c.shape[1]

    for _ in range(n):
        eigenvalue, eigenvector = dominant_eig(A_c, max_iterations)

        if np.linalg.norm(eigenvector) < 1e-12 or np.isnan(eigenvalue):
            continue

        eigenvalues.append(eigenvalue)

        eigenvector = eigenvector / norm(eigenvector)
        for v in eigenvectors:
            eigenvector -= np.dot(v, eigenvector) * v
        eigenvectors.append(eigenvector)
        A_c = A_c - eigenvalue * np.outer(eigenvector, eigenvector)

    while len(eigenvalues) < n:
        eigenvalues.append(0.0)
        eigenvectors.append(np.zeros(n))

    return np.array(eigenvalues), np.array(eigenvectors)


def svd(A: np.ndarray, full_matrices: bool = True, tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the SVD of a matrix using eigen decomposition.

    :param A: Input matrix of shape (m, n).
    :param full_matrices: Whether to compute full-sized U and V matrices.
    :return: Tuple (U, D, V_T), where A ≈ U @ np.diag(D) @ V_T.
    '''
    m, n = A.shape
    k = min(m, n)

    U = np.zeros((m, m if full_matrices else k))
    V = np.zeros((n, n if full_matrices else k))

    eigenvalues, eigenvectors = eig(np.dot(A.T, A))

    eigenvectors = np.where(np.abs(eigenvectors) < tolerance, 0.0, eigenvectors)
    eigenvalues = eigenvalues[:k]
    eigenvectors = eigenvectors[:k]
    eigenvalues = np.clip(eigenvalues, 0, None)
    D = np.sqrt(eigenvalues)
    D = D[(idx := np.argsort(-D))]
    eigenvectors = eigenvectors[idx]

    for i, eigenvector in enumerate(eigenvectors):
        V[:, i] = eigenvector
        if D[i] > tolerance:
            U[:, i] = np.dot(A, eigenvector) / D[i]
        else:
            U[:, i] = 0

    if full_matrices:
        if m > k:
            U_full = np.random.rand(m, m - k)
            U_full, _ = gram_schmidt_qr(U_full)
            U_full = U_full[:, :m - k]
            U[:, k:] = U_full
        elif n > k:
            V_full = np.random.rand(n, n - k)
            V_full, _ = gram_schmidt_qr(V_full)
            V_full = V_full[:, :n - k]
            V[:, k:] = V_full

    V, _ = gram_schmidt_qr(V)
    return U, D, V.T
