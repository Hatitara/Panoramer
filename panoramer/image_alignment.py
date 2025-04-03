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




def inverse_homography(H: np.ndarray) -> np.ndarray:
    """
    Manually computes the inverse of a 3x3 homography matrix using the determinant and cofactors.

    :param H: A 3x3 homography matrix
    :return: The manually computed inverse of H
    """
    h11, h12, h13 = H[0, :]
    h21, h22, h23 = H[1, :]
    h31, h32, h33 = H[2, :]

    # Compute the determinant of H
    det_H = (h11 * (h22 * h33 - h23 * h32)
           - h12 * (h21 * h33 - h23 * h31)
           + h13 * (h21 * h32 - h22 * h31))

    if abs(det_H) < 1e-6:  # Avoid division by zero
        raise ValueError("Matrix is singular and cannot be inverted.")

    # Compute the cofactor matrix
    C = np.array([
        [ (h22 * h33 - h23 * h32), -(h12 * h33 - h13 * h32),  (h12 * h23 - h13 * h22)],
        [-(h21 * h33 - h23 * h31),  (h11 * h33 - h13 * h31), -(h11 * h23 - h13 * h21)],
        [ (h21 * h32 - h22 * h31), -(h11 * h32 - h12 * h31),  (h11 * h22 - h12 * h21)]
    ])

    # Compute the adjugate (transpose of the cofactor matrix)
    adj_H = C.T

    # Compute the inverse: adjugate / determinant
    H_inv = (1 / det_H) * adj_H

    return H_inv






def compute_homography_svd(src_pts, dst_pts):
    """
    Computes the homography matrix using SVD.
    :param src_pts: Source points (Nx2)
    :param dst_pts: Destination points (Nx2)
    :return: 3x3 Homography matrix
    """
    assert src_pts.shape[0] == dst_pts.shape[0] and src_pts.shape[0] >= 4, "At least 4 points are required"

    num_points = src_pts.shape[0]
    A = []

    for i in range(num_points):
        x, y = src_pts[i]
        x_prime, y_prime = dst_pts[i]

        A.append([-x, -y, -1,  0,  0,  0,  x * x_prime, y * x_prime, x_prime])
        A.append([ 0,  0,  0, -x, -y, -1,  x * y_prime, y * y_prime, y_prime])

    A = np.array(A)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(A)
    
    # Homography matrix is the last column of V (or last row of V transposed)
    H = Vt[-1].reshape(3, 3)
    
    return H / H[2, 2]  # Normalize so that H[2,2] = 1



def warp_perspective_manual(image: np.ndarray, H: np.ndarray, output_shape):
    w_out, h_out = output_shape
    warped_image = np.zeros((h_out, w_out, 3), dtype=np.uint8)

    # Обчислюємо обернену матрицю гомографії
    H_inv = np.linalg.inv(H)

    for y_out in range(h_out):
        for x_out in range(w_out):
            # Трансформуємо вихідні координати у вхідні
            vec = np.array([x_out, y_out, 1])
            x_in, y_in, w = H_inv @ vec
            x_in /= w
            y_in /= w

            # Використовуємо білярну інтерполяцію
            if 0 <= x_in < image.shape[1] - 1 and 0 <= y_in < image.shape[0] - 1:
                x0, y0 = int(x_in), int(y_in)
                dx, dy = x_in - x0, y_in - y0

                # Чотири сусідні пікселі
                top_left = image[y0, x0].astype(float)
                top_right = image[y0, x0 + 1].astype(float)
                bottom_left = image[y0 + 1, x0].astype(float)
                bottom_right = image[y0 + 1, x0 + 1].astype(float)

                # Білярна інтерполяція
                top = (1 - dx) * top_left + dx * top_right
                bottom = (1 - dx) * bottom_left + dx * bottom_right
                interpolated = (1 - dy) * top + dy * bottom

                warped_image[y_out, x_out] = interpolated.astype(np.uint8)

    return warped_image





def warp_image(homography: np.ndarray, sec_img: np.ndarray, first_img: np.ndarray) -> np.ndarray:
    """
    Warps an image based on the given homography matrix using SVD.
    """
    sec_img_shape = sec_img.shape[:2]
    first_img_shape = first_img.shape[:2]
    Height, Width = sec_img_shape

    # Define the initial corner points of the secondary image
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])

    # Compute the new corner positions
    FinalMatrix = np.dot(homography, InitialMatrix)
    FinalMatrix /= FinalMatrix[2]  # Normalize the homogeneous coordinates

    x, y = FinalMatrix[:2]
    
    # Determine the new dimensions of the stitched image
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width, New_Height = max_x, max_y
    Correction = [0, 0]
    
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)

    # Ensure the stitched image is large enough to fit the base image
    New_Width = max(New_Width, first_img_shape[1] + Correction[0])
    New_Height = max(New_Height, first_img_shape[0] + Correction[1])

    # Shift coordinates to positive space
    x += Correction[0]
    y += Correction[1]
    
    OldInitialPoints = np.float32([[0, 0], [Width - 1, 0], [Width - 1, Height - 1], [0, Height - 1]])
    NewFinalPoints = np.float32(np.array([x, y]).T)

    # Compute the corrected homography matrix using SVD
    # HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPoints)
    HomographyMatrix = compute_homography_svd(OldInitialPoints, NewFinalPoints)

    # Warp the image
    
    StitchedImage = cv2.warpPerspective(first_img, HomographyMatrix, (New_Width, New_Height))

    StitchedImage[Correction[1]:Correction[1] + sec_img_shape[0], Correction[0]:Correction[0] + sec_img_shape[1]] = sec_img
    
    return StitchedImage