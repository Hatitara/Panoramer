'''
Module for visualization functions.
'''
import cv2
import matplotlib.pyplot as plt


def show_image_jupyter(img: cv2.typing.MatLike) -> None:
    '''
    Visualising an image in Jupyter notebook.
    '''
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def show_image_new_window(img: cv2.typing.MatLike):
    '''
    Visualising an image in the new window.
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)


def show_panorama_jupyter(images: list, panorama: cv2.typing.MatLike) -> None:
    '''
    Displays individual images alongside the final panorama in Jupyter notebook.
    '''
    pass


def show_panorama_new_window(images: list, panorama: cv2.typing.MatLike) -> None:
    '''
    Displays individual images alongside the final panorama in the new window.
    '''
    pass
