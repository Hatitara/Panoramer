'''
Used to align images based on the detected features.
'''
import cv2
import numpy as np
import random

def compute_homography_dlt(points1: np.ndarray, points2: np.ndarray):
    '''
    Computes homography using Direct Linear Transform (DLT) method.
    
    :param points1: Array of shape (N, 2) with keypoints from image 1.
    :param points2: Array of shape (N, 2) with keypoints from image 2.
    :return: 3x3 Homography matrix.
    '''
    if len(points1) < 4:
        return None

    A = []
    for i in range(len(points1)):
        x, y = points1[i]
        u, v = points2[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

    A = np.array(A)

    _, _, Vt = np.linalg.svd(A) # REWRITE FOR CUSTOM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    H = Vt[-1].reshape(3, 3)

    return H / H[2, 2]


def apply_homography(points: np.ndarray, H: np.ndarray):
    '''
    Applies a homography transformation to a set of points.
    
    :param points: Array of shape (N, 2).
    :param H: 3x3 Homography matrix.
    :return: Transformed points of shape (N, 2).
    '''
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = (H @ points_homo.T).T
    transformed_points /= transformed_points[:, 2].reshape(-1, 1)
    return transformed_points[:, :2]  # Return (x, y) coordinates


def ransac_homography(points1: np.ndarray, points2: np.ndarray, threshold: float = 3.0, iterations: int = 1000):
    '''
    Computes a robust homography matrix using RANSAC.
    
    :param points1: Array of shape (N, 2) with keypoints from image 1.
    :param points2: Array of shape (N, 2) with keypoints from image 2.
    :param threshold: Maximum reprojection error to consider a match as an inlier.
    :param iterations: Number of RANSAC iterations.
    :return: Best homography matrix and inlier mask.
    '''
    max_inliers = []
    best_H = None

    assert len(points1) == len(points2), "Mismatched keypoint array sizes"

    for _ in range(iterations):
        sample_indices = random.sample(range(len(points1)), 4)
        pts1_sample = points1[sample_indices]
        pts2_sample = points2[sample_indices]

        H = compute_homography_dlt(pts1_sample, pts2_sample)
        if H is None:
            continue

        transformed_pts = apply_homography(points1, H)
        errors = np.linalg.norm(transformed_pts - points2, axis=1)
        inliers = np.where(errors < threshold)[0]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_H = H

    if best_H is not None and len(max_inliers) > 4:
        best_H = compute_homography_dlt(points1[max_inliers], points2[max_inliers])

    return best_H, max_inliers


def find_homography(points1: np.ndarray, points2: np.ndarray, method = None) -> np.ndarray:
    '''
    Computes a robust homography matrix using RANSAC.
    
    :param points1: Array of shape (N, 2) with keypoints from image 1.
    :param points2: Array of shape (N, 2) with keypoints from image 2.
    :return: Homography matrix.
    '''
    if method is None:
        H, _ = ransac_homography(points1, points2)
    elif method == 'RANSAC':
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    elif method == 'LMEDS':
        H, _ = cv2.findHomography(points1, points2, cv2.LMEDS)
    elif method == 'RHO':
        H, _ = cv2.findHomography(points1, points2, cv2.RHO)
    else:
        raise ValueError(f"Unknown homography method: {method}")
    return H


def warp_image(image: cv2.typing.MatLike, homography: np.ndarray) -> cv2.typing.MatLike:
    '''
    Warps an image based on the given homography matrix.
    '''
    pass
