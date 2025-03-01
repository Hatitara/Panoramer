'''
Module for visualization functions.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image_jupyter(img: cv2.typing.MatLike) -> None:
    '''
    Visualising an image in Jupyter notebook.
    '''
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def show_image_new_window(img: cv2.typing.MatLike):
    '''
    Visualising an image in the new window.
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)


def visualize_keypoints_jupyter(image: np.ndarray, keypoints: np.ndarray) -> None:
    '''
    Plots an image with detected keypoints overlayed.

    :param image: The input image (BGR format).
    :param keypoints: Array of detected keypoints as (x, y) coordinates.
    '''
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=10, label="Keypoints")
    plt.legend()
    plt.title("Detected Keypoints")
    plt.axis("off")
    plt.show()


def visualize_matches(image1: cv2.typing.MatLike, image2: cv2.typing.MatLike, 
                      keypoints1: np.ndarray, keypoints2: np.ndarray, matches: list):
    '''
    Visualizes feature matches between two images.

    :param image1: First input image.
    :param image2: Second input image.
    :param keypoints1: Keypoints detected in the first image (N,2 array).
    :param keypoints2: Keypoints detected in the second image (N,2 array).
    :param matches: List of matched keypoint indices.
    '''
    kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints1]
    kp2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints2]

    cv_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0) for i, j in matches]
    match_img = cv2.drawMatches(image1, kp1, image2, kp2, cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def show_panorama_jupyter(images: list, panorama: cv2.typing.MatLike) -> None:
    '''
    Displays individual images alongside the final panorama in Jupyter notebook.
    '''
    pass


def show_panorama_new_window(images: list, panorama: cv2.typing.MatLike) -> None:
    '''
    Displays individual images alongside the final panorama in the new window.
    '''
    pass
