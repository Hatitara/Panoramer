'''
Used to align images based on the detected features.
'''
import cv2
import numpy as np


def find_homography(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    '''
    Computes the homography matrix to align two sets of points.
    '''
    pass


def warp_image(image: cv2.typing.MatLike, homography: np.ndarray) -> cv2.typing.MatLike:
    '''
    Warps an image based on the given homography matrix.
    '''
    pass
