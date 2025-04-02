'''
Module for extracting key features (corners, edges, etc.) from images for alignment
'''
import cv2
import numpy as np

from .utils import apply_gaussian_blur, apply_sobel


def detect_features(image: cv2.typing.MatLike, method: str = None, k: float = 0.04) -> list:
    '''
    Detects keypoints in an image using either a custom Harris Corner Detector or a built-in OpenCV method.

    :param image: The input image.
    :param method: The feature detection method to use ('ORB', 'SIFT', 'SURF'). If None, uses a custom method.
    :return: An array of detected keypoints.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method:
        if method.upper() == 'SIFT':
            sift = cv2.SIFT_create()
            keypoints = sift.detect(gray, None)
        elif method.upper() == 'ORB':
            orb = cv2.ORB_create()
            keypoints = orb.detect(gray, None)
        elif method.upper() == 'SURF':
            try:
                surf = cv2.xfeatures2d.SURF_create()
                keypoints = surf.detect(gray, None)
            except AttributeError:
                raise ValueError(
                    "SURF is not available in this OpenCV build. Try installing OpenCV-contrib.")
        else:
            raise ValueError(f"Unsupported method: {method}")
        return np.array([kp.pt for kp in keypoints])

    # === Custom Harris Corner Detection ===
    blurred = apply_gaussian_blur(gray, kernel_size=3, sigma=1.0)

    Ix = apply_sobel(blurred, direction='x')
    Iy = apply_sobel(blurred, direction='y')

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    Ixx = apply_gaussian_blur(Ixx, kernel_size=3, sigma=1.0)
    Iyy = apply_gaussian_blur(Iyy, kernel_size=3, sigma=1.0)
    Ixy = apply_gaussian_blur(Ixy, kernel_size=3, sigma=1.0)

    det_M = (Ixx * Iyy) - (Ixy ** 2)
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M ** 2)

    threshold = 0.01 * R.max()
    keypoints = np.argwhere(R > threshold)

    return keypoints[:, ::-1]


def wcompute_descriptors(image: cv2.typing.MatLike, keypoints: np.ndarray, method: str = None, patch_size: int = 16) -> np.ndarray:
    '''
    Computes descriptors for the detected keypoints.

    :param image: The input image.
    :param keypoints: NumPy array of shape (N, 2) containing keypoint coordinates.
    :param method: Feature extraction method ('SIFT', 'ORB', 'SURF') or None for custom implementation.
    :return: Computed descriptors as a NumPy array.
    '''
    if method == "SIFT":
        sift = cv2.SIFT_create()
        keypoints_cv2 = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints]
        _, descriptors = sift.compute(image, keypoints_cv2)
    elif method == "ORB":
        orb = cv2.ORB_create()
        keypoints_cv2 = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints]
        _, descriptors = orb.compute(image, keypoints_cv2)
    elif method == "SURF":
        surf = cv2.xfeatures2d.SURF_create()
        keypoints_cv2 = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints]
        _, descriptors = surf.compute(image, keypoints_cv2)
    else:
        descriptors = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            half_size = patch_size // 2

            patch = gray[max(0, y - half_size):min(gray.shape[0], y + half_size),
                         max(0, x - half_size):min(gray.shape[1], x + half_size)]

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))

            descriptor = patch.flatten().astype(np.float32)
            descriptor /= np.linalg.norm(descriptor) if np.linalg.norm(descriptor) > 0 else 1
            descriptors.append(descriptor)

        descriptors = np.array(descriptors, dtype=np.float32)

    return descriptors


def apply_lowe_ratio_test(matches, ratio=0.75):
    '''
    Applies Lowe's Ratio Test to filter out weak matches.

    :param matches: List of knn matches (pairs of nearest and second nearest matches).
    :param ratio: The ratio threshold (default: 0.75).
    :return: Filtered list of good matches.
    '''
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray, method: str = None, ratio: float = 0.6) -> list:
    '''
    Matches features between two images using the specified method.

    :param image1: First input image.
    :param image2: Second input image.
    :param keypoints1: NumPy array of shape (N, 2) containing keypoint coordinates in the first image.
    :param keypoints2: NumPy array of shape (M, 2) containing keypoint coordinates in the second image.
    :param descriptors1: Descriptors computed for keypoints in the first image.
    :param descriptors2: Descriptors computed for keypoints in the second image.
    :param method: Feature matching method ('BF', 'FLANN') or None for custom matching.
    :return: List of matched keypoint index pairs.
    '''
    matches = []
    if method == "BF":
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        matches = apply_lowe_ratio_test(matches, ratio=ratio)
    elif method == "FLANN":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        matches = apply_lowe_ratio_test(matches, ratio=ratio)
    else:
        # Custom feature matching using Nearest Neighbor search
        for i, desc1 in enumerate(descriptors1):
            distances = np.linalg.norm(descriptors2 - desc1, axis=1)
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
            distances[best_idx] = np.inf  # Temporarily remove best match
            second_best_distance = np.min(distances)

            if best_distance < ratio * second_best_distance:
                matches.append((i, best_idx))

    return matches
