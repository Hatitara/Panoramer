'''
Module for combining aligned images into a panorama.
'''
import cv2
import numpy as np

import panoramer as pano
def stitch_images(images: list[cv2.typing.MatLike], built_in_warper: bool = False) -> cv2.typing.MatLike:
    '''
    Merges multiple aligned images into a single panorama.
    '''
    prev_image = images[0]
    
    for image in images[1:]:
        prev_image_resized = pano.resize_image(prev_image, width=400, height=None)
        cur_image_resized = pano.resize_image(image, width=400, height=None)
        prev_image_normalized = pano.normalize_brightness_contrast(prev_image_resized)
        cur_image_normalized = pano.normalize_brightness_contrast(cur_image_resized)
        
        prev_image_keypoints = pano.detect_features(prev_image_normalized)
        print(f"Found {len(prev_image_keypoints)} keypoints in previous image")
        cur_image_keypoints = pano.detect_features(cur_image_normalized)
        print(f"Found {len(cur_image_keypoints)} keypoints in current image")
        
        prev_image_descriptors = pano.compute_descriptors(prev_image_normalized, prev_image_keypoints)
        print(f"Computed {len(prev_image_descriptors)} for previous image")
        cur_image_descriptors = pano.compute_descriptors(cur_image_normalized, cur_image_keypoints)
        print(f"Computed {len(cur_image_descriptors)} for current image")
        
        matches = pano.match_features(prev_image_descriptors, cur_image_descriptors, method="RANSAC")
        print(f"Found {len(matches)} matches")

        if len(matches) < 4:
            raise ValueError("Not enough matches to compute homography!")
        
        matched_points1 = np.array([prev_image_keypoints[m[0]] for m in matches], dtype=np.float32)
        matched_points2 = np.array([cur_image_keypoints[m[1]] for m in matches], dtype=np.float32)

        homography = pano.find_homography(matched_points1, matched_points2)
        warped_image = pano.warp_image(homography, cur_image_resized, prev_image_resized, built_in_warper=built_in_warper)
        prev_image_normalized = warped_image
    
    return warped_image
