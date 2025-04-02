'''
Module for combining aligned images into a panorama.
'''
import cv2
import numpy as np

import panoramer as pano

def stitch_images(images: list[cv2.typing.MatLike]) -> cv2.typing.MatLike:
    '''
    Merges multiple aligned images into a single panorama.
    '''
    prev_image = images[0]
    prev_image_resized = pano.resize_image(prev_image, width=400, height=None)
    prev_image_normalized = pano.normalize_brightness_contrast(prev_image_resized)
    for image in images[1:]:
        pano.show_panorama_jupyter(prev_image)
        prev_image_resized = pano.resize_image(prev_image, width=400, height=None)
        cur_image_resized = pano.resize_image(image, width=400, height=None)
        
        cur_image_normalized = pano.normalize_brightness_contrast(cur_image_resized)
        
        prev_image_keypoints = pano.detect_features(prev_image_normalized)
        cur_image_keypoints = pano.detect_features(cur_image_normalized)
        
        prev_image_descriptors = pano.compute_descriptors(prev_image_normalized, prev_image_keypoints)
        cur_image_descriptors = pano.compute_descriptors(cur_image_normalized, cur_image_keypoints)
        
        matches = pano.match_features(prev_image_descriptors, cur_image_descriptors, method="RANSAC")
        
        matched_points1 = np.array([prev_image_keypoints[m[0]] for m in matches], dtype=np.float32)
        matched_points2 = np.array([cur_image_keypoints[m[1]] for m in matches], dtype=np.float32)

        print(f"Found {len(matches)} matches")

        if len(matches) < 4:
            raise ValueError("Not enough matches to compute homography!")

        homography = pano.find_homography(matched_points1, matched_points2)
        warped_image = pano.warp_image(homography, cur_image_normalized, prev_image_normalized)
        prev_image = warped_image
    
    return prev_image
