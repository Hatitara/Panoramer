'''
Used to align images based on the detected features.
'''
import cv2
import numpy as np
import random
from .utils import inverse_homography, svd


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

    _, _, Vt = svd(A)
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
    return transformed_points[:, :2]


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


def warp_perspective(image: np.ndarray, H: np.ndarray, output_shape):
    '''
    Applies a manual perspective warp to an image using a given homography matrix.

    :param image: Input image as a NumPy array.
    :param H: 3x3 Homography matrix for the perspective transformation.
    :param output_shape: Tuple (width, height) specifying the dimensions of the output image.
    :return: Warped image as a NumPy array with the specified output dimensions.
    '''

    w_out, h_out = output_shape
    warped_image = np.zeros((h_out, w_out, 3), dtype=np.uint8)

    H_inv = inverse_homography(H)

    for y_out in range(h_out):
        for x_out in range(w_out):
            vec = np.array([x_out, y_out, 1])
            x_in, y_in, w = H_inv @ vec
            x_in /= w
            y_in /= w

            if 0 <= x_in < image.shape[1] - 1 and 0 <= y_in < image.shape[0] - 1:
                x0, y0 = int(x_in), int(y_in)
                dx, dy = x_in - x0, y_in - y0

                top_left = image[y0, x0].astype(float)
                top_right = image[y0, x0 + 1].astype(float)
                bottom_left = image[y0 + 1, x0].astype(float)
                bottom_right = image[y0 + 1, x0 + 1].astype(float)

                top = (1 - dx) * top_left + dx * top_right
                bottom = (1 - dx) * bottom_left + dx * bottom_right
                interpolated = (1 - dy) * top + dy * bottom

                warped_image[y_out, x_out] = interpolated.astype(np.uint8)

    return warped_image





def warp_image(homography: np.ndarray, sec_img: np.ndarray, first_img: np.ndarray, built_in_warper=False) -> np.ndarray:
    '''
    Warps an image based on the given homography matrix using DLT.
    '''
    sec_img_shape = sec_img.shape[:2]
    first_img_shape = first_img.shape[:2]
    h, w = sec_img_shape

    init_matrix = np.array([[0, w - 1, w - 1, 0],
                            [0, 0, h - 1, h - 1],
                            [1, 1, 1, 1]])

    final_matrix = homography @ init_matrix
    final_matrix /= final_matrix[2]

    x, y = final_matrix[:2]
    
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    new_w, new_h = max_x, max_y
    corrections = [0, 0]
    
    if min_x < 0:
        new_w -= min_x
        corrections[0] = abs(min_x)
    if min_y < 0:
        new_h -= min_y
        corrections[1] = abs(min_y)

    new_w = max(new_w, first_img_shape[1] + corrections[0])
    new_h = max(new_h, first_img_shape[0] + corrections[1])

    x += corrections[0]
    y += corrections[1]
    
    old_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    new_points = np.float32(np.array([x, y]).T)

    H = compute_homography_dlt(old_points, new_points)

    stiched_img = cv2.warpPerspective(first_img, H, (new_w, new_h)) if built_in_warper else warp_perspective(first_img, H, (new_w, new_h))

    y_end = min(corrections[1] + sec_img_shape[0], stiched_img.shape[0])
    x_end = min(corrections[0] + sec_img_shape[1], stiched_img.shape[1])

    y_crop = y_end - corrections[1]
    x_crop = x_end - corrections[0]

    stiched_img[corrections[1]:y_end, corrections[0]:x_end] = sec_img[:y_crop, :x_crop]

    return stiched_img