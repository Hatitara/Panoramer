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
    for image in images[1:]:
        
        prev_image_resized = pano.resize_image(prev_image, width=400, height=None)
        cur_image_resized = pano.resize_image(image, width=400, height=None)
        
        prev_image_keypoints = pano.detect_features(prev_image_resized)
        cur_image_keypoints = pano.detect_features(cur_image_resized)
        
        prev_image_descriptors = pano.compute_descriptors(prev_image_resized, prev_image_keypoints)
        cur_image_descriptors = pano.compute_descriptors(cur_image_resized, cur_image_keypoints)
        
        matches = pano.match_features(prev_image_descriptors, cur_image_descriptors, method="RANSAC")
        
        homography = pano.find_homography(matches[0], matches[1], method='RANSAC')
        
        warped_image = pano.warp_image(cur_image_resized, homography)
        
        prev_image = pano.(prev_image_resized, warped_image)
    
    # normalized_images = [pano.normalize_brightness_contrast(image, None, 0, 255, cv2.NORM_MINMAX) for image in resized_images]
    
    # keypoints = [pano.detect_features(image) for image in normalized_images]
    
    # descriptors = [pano.compute_descriptors(image, keypoints) for image in normalized_images]
    
    # matches = [pano.match_features(image1, image2) for image1, image2 in zip(normalized_images, normalized_images[1:])]
    
    # homographies = [pano.find_homography(match[0], match[1],  method='RANSAC') for match in matches]
    
    # warped_images = [pano.warp_image(image, homography) for image, homography in zip(normalized_images, homographies)]
    
    # panorama = pano.stitch_images(warped_images)
    
    return panorama
