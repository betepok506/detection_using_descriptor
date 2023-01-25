import cv2 as cv
import numpy as np


def read_img(file_path: str):
    '''
    Функция для чтения изображений, содержащих в пути unicode символы

    Parameter
    -----------
    file_path: `str`
        Путь до изображения

    Returns
    -----------
    bgr_img: `ndarray` or `None`
        Считаное изображение в формате BGR
    '''
    # TODO: Протестировать все случаи чтения изображения и обработки ошибок
    stream = open(file_path, 'rb')
    bytes = bytearray(stream.read())
    if len(bytes) == 0:
        return None
    array = np.asarray(bytes, dtype=np.uint8)
    bgr_img = cv.imdecode(array, cv.IMREAD_UNCHANGED)
    return bgr_img
