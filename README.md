# LA Project: `Panoramer` (Panorama Merging Project)

## To Do:
- [ ] Do the research about already realized similar projects
- [x] Get some images to work with
- [ ] Research general idea of how to do this
- [ ] Report №1
- [ ] Report №2
- [ ] Report №3
- [ ] Final Report

## Project Overview
This project aims to merge multiple photos into a seamless panorama using Python and OpenCV. The goal is to automate the process of stitching together multiple images, taken from similar angles or positions, into a single panoramic image. The package will consist of various modules that handle image loading, feature detection, image alignment, stitching, and visualization.

## Project Structure
The project is divided into the following modules:
1. **image_loader.py**
    - Purpose: Handles the loading and saving of images.
    - Functions: 
        - `load_image(filepath: str) -> cv2.typing.MatLike`: Loads an image from a given file path.
        - `save_image(image: cv2.typing.MatLike, output_path: str) -> None`: Saves an image to a specified output path.
2. **feature_extraction.py**
    - Purpose: Extracts key features and computes descriptors from images.
    - Functions: 
        - `detect_features(image: cv2.typing.MatLike) -> list`: Detects key points in the image.
        - `compute_descriptors(image: cv2.typing.MatLike, keypoints: list) -> np.ndarray`: Computes descriptors for the detected key points.
        - `match_features(image1: cv2.typing.MatLike, image2: cv2.typing.MatLike) -> list`: Matches features between two images.
3. **image_alignment.py**
    - Purpose: Aligns images based on their features.
    - Functions: 
        - `find_homography(points1: np.ndarray, points2: np.ndarray) -> np.ndarray`: Computes the homography matrix to align two sets of points.
        - `warp_image(image: cv2.typing.MatLike, homography: np.ndarray) -> cv2.typing.MatLike`: Warps an image based on the given homography matrix.
4. **panorama.py**
    - Purpose: Merges aligned images into a single panorama.
    - Functions: 
        - `stitch_images(images: list) -> cv2.typing.MatLike`: Merges multiple aligned images into a single panorama.
5. **visualization.py**
    - Purpose: Displays images and the final panorama.
    - Functions: 
        - `show_image(image: cv2.typing.MatLike) -> None`: Displays a single image.
        - `show_panorama(images: list, panorama: cv2.typing.MatLike) -> None`: Displays individual images alongside the final panorama.
6. **utils.py**
    - Purpose: Provides helper functions for image processing tasks.
    - Functions: 
        - `resize_image(image: cv2.typing.MatLike, width: int, height: int) -> cv2.typing.MatLike`: Resizes an image to the specified width and height.
        - `blend_images(image1: cv2.typing.MatLike, image2: cv2.typing.MatLike) -> cv2.typing.MatLike`: Blends two images together to create a seamless transition.

## Steps to Merge Images
1. Preprocessing
    - Resize images to the same scale if needed.
    - Normalize brightness/contrast across images to ensure consistency.
2. Feature Detection & Matching
    - Detect features in images using methods like ORB, SIFT, or SURF.
    - Match corresponding features between images to find points of alignment.
3. Homography Estimation
    - Compute a homography matrix using matched points to align images correctly.
4. Image Warping
    - Warp images using the homography matrix to align them in the same coordinate system.
5. Image Stitching
    - Combine the warped images into a single panorama.
    - Optionally, apply advanced blending techniques for seamless merging.
6. Post-processing
    - Crop the edges to remove unwanted areas.
    - Apply color correction or further blending if needed for a smooth result.

## Dependencies
- *Python 3.x*
- *OpenCV* (`opencv-python`)
- *NumPy* (`numpy`)
- *Matplotlib* (`matplotlib`) for visualization in `Jupyter` notebook

## Possible future enhancements
- Support for automatic detection of image overlap and stitching regions.
- Integration of advanced blending techniques such as multi-band blending or seam carving.
- Command-line interface (`CLI`) for easy usage.


## References
- Images taken from [Panorama](https://github.com/DatCanCode/panorama) by [DatCanCode](https://github.com/DatCanCode)