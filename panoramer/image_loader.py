'''
Handle loading and saving images.
'''
import cv2


def load_image(path: str) -> cv2.typing.MatLike:
    '''
    Reads an image in CV2 format.
    '''
    return cv2.imread(path)


def save_image(image: cv2.typing.MatLike, output_path: str) -> None:
    '''
    Saves the image to a specified file path.
    '''
    cv2.imwrite(output_path, image)
