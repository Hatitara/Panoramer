'''
Module for extracting key features (corners, edges, etc.) from images for alignment
'''
import cv2
import numpy as np

def detect_features(image: cv2.typing.MatLike) -> list:
    '''
    Detects key points in the image.
    '''
    pass

def compute_descriptors(image: cv2.typing.MatLike, keypoints: list) -> np.ndarray:
    '''
    Computes descriptors for the detected key points.
    '''
    pass

def match_features(image1: cv2.typing.MatLike, image2: cv2.typing.MatLike) -> list:
    '''
    Matches features between two images.
    '''
    pass
