import os
from typing import Tuple, NoReturn
import pybboxes as pbx
from src.utils.utils import read_img
from src.evaluation.bounding_box import BoundingBox
from src.evaluation.bounding_boxes import BoundingBoxes
from src.utils.utils_evaluator import *


class DatasetFormat():
    YOLO_DARKNET: str = "YOLO Darknet"
    CLASSIFICATION: str = "Classification"


class BaseReader():
    def _reading_img(self, path_to_image: str):
        '''
        Функция для чтения изображения по заданному пути

        Parameter
        -----------
        path_to_image: `str`
            Путь до считываемого изображения

        Returns
        -----------
        `np.array`
            Считанное изображение
        '''
        return read_img(path_to_image)


class ClassificationReader(BaseReader):
    def __init__(self, path_to_folder_with_images):
        self.path_to_folder_with_datasets = path_to_folder_with_images

    def get_item(self, cls_folder: str, img_name: str):
        path_to_img = os.path.join(self.path_to_folder_with_datasets,
                                   cls_folder,
                                   img_name)
        img = read_img(path_to_img)
        if img is None:
            return None

        bboxes = self._reading_annotation(cls_folder, img.shape, img_name)
        item_dataset = ImagesDatasetItem(img=img,
                                         img_name=img_name,
                                         bboxes=bboxes)
        return item_dataset

    def _reading_annotation(self,
                            cls: str,
                            img_shape: Tuple[int, int],
                            img_name: str):
        '''
        Функция возвращает аннотация изображения как одного Bbox. Размеры Bbox
        соответственно равны размеру изображения

        Parameter
        -----------
        cls: `str`
            Наименование класса
        img_shape: `Tuple[int, int]`
            Размеры изображения в формате (H, W)
        img_name: `str`
            Наименование изображения

        Returns:
        -----------
        `BoundingBoxes`
            Аннотация изображения в виде одного `BoundingBox`, содержащего информацию о избражении
        '''
        all_bounding_boxes = BoundingBoxes()
        bbox = BoundingBox(
            imageName=img_name,
            classId=cls,
            x=0, y=0,
            w=img_shape[0], h=img_shape[1],
            imgSize=img_shape,
            bbType=BBType.GroundTruth,
            format=BBFormat.XYX2Y2
        )
        all_bounding_boxes.addBoundingBox(bbox)

        return all_bounding_boxes

    def __iter__(self):
        for cls_folder in os.listdir(self.path_to_folder_with_datasets):
            path_to_folder_with_images = os.path.join(self.path_to_folder_with_datasets,
                                                      cls_folder)
            if not os.path.isdir(path_to_folder_with_images):
                continue

            for img_name in os.listdir(path_to_folder_with_images):
                yield self.get_item(cls_folder, img_name)


class YoloDarknetReader(BaseReader):
    '''
    Класс, реализующий взаимодействие с датасетом в формате YOLO Darknet
    '''

    def __init__(self, path_to_folder_with_images: str,
                 path_to_folder_with_labels: str):
        self.path_to_folder_with_images = path_to_folder_with_images
        self.path_to_folder_with_labels = path_to_folder_with_labels
        self._format_file_with_annotation = ".txt"
        self._annotation_separator = " "

    def _reading_annotation(self, path_to_file_with_annotation: str,
                            img_shape: Tuple[int, int],
                            img_name: str):
        '''
        Функция для чтения файла с аннотацией и преобразования его в VOC формат

        Parameter
        -----------
        path_to_file_with_annotation: `str`
            Путь до считываемого файла с аннотацией
        img_shape: `Tuple[int, int]`
            Размер изображения в формате (height, width)
        img_name: `str`
            Имя изображения

        Returns
        -----------
        `BoundingBoxes`
            Класс-контейнер, содержащий набор BoundingBox
        '''
        if not os.path.exists(path_to_file_with_annotation):
            return None

        all_bounding_boxes = BoundingBoxes()
        with open(path_to_file_with_annotation, "r") as file:
            for line in file:
                elements = line.rstrip().split(self._annotation_separator)
                if len(elements) != 5:
                    raise "The file does not match the Yolo Darknet format!"

                class_id = elements[0]
                box = list(map(float, elements[1:]))
                h, w, _ = img_shape
                try:
                    x, y, x2, y2 = pbx.convert_bbox(box, from_type="yolo", to_type="voc", image_size=(w, h))
                except Exception as e:
                    print(e)
                    return None

                bbox = BoundingBox(
                    imageName=img_name,
                    classId=class_id,
                    x=x, y=y,
                    w=x2, h=y2,
                    imgSize=img_shape,
                    bbType=BBType.GroundTruth,
                    format=BBFormat.XYX2Y2
                )
                all_bounding_boxes.addBoundingBox(bbox)

        return all_bounding_boxes

    def get_item(self, img_name: str):
        '''
        Функция позволяет получить по имени изображения Bbox и само изображение

        Parameter
        -----------
        img_name: `str`
            Имя изображения

        Returns
        -----------
        `ItemDataset`
            Информация о считанном файле
        '''
        name_file = os.path.splitext(img_name)[0]
        path_to_img = os.path.join(self.path_to_folder_with_images,
                                   img_name)
        path_to_annotation = os.path.join(self.path_to_folder_with_labels,
                                          name_file + self._format_file_with_annotation)
        img = self._reading_img(path_to_img)
        if img is None:
            return None

        bboxes = self._reading_annotation(path_to_annotation,
                                          img.shape,
                                          img_name)
        if bboxes is None:
            return None

        item_dataset = ImagesDatasetItem(img=img,
                                         img_name=img_name,
                                         bboxes=bboxes)
        return item_dataset

    def __iter__(self):
        for elem in os.listdir(self.path_to_folder_with_images):
            result = self.get_item(elem)
            if result is None:
                continue

            yield result


class ImagesDatasetItem():
    def __init__(self, img, img_name, bboxes):
        self.img = img
        self.img_name = img_name
        self.bboxes = bboxes


class ImagesDataset():
    def __init__(self, path_to_folder_with_images: str,
                 path_to_folder_with_labels: str = None,
                 dataset_format: str = DatasetFormat.CLASSIFICATION):
        self.path_to_folder_with_images = path_to_folder_with_images
        self.path_to_folder_with_labels = path_to_folder_with_labels
        self.dataset_reader = self._get_reader(dataset_format)

    def read(self, img_name):
        '''
        Функция позволяет по имени изображения получить Bbox и само изображение

        Parameter
        -----------
        img_name: `str`
            Имя изображения

        Returns
        -----------
        `ItemDataset`
            Информация о считанном файле
        '''
        return self.dataset_reader.get_item(img_name)

    def _get_reader(self, dataset_format):
        '''
        Функция для получения нужного класса для чтения

        Parameter
        -----------
        dataset_format: `str`
            Формат данных

        Returns
        -----------
        `YoloDarknetReader`
            Класс для считывания данных
        '''
        if dataset_format == DatasetFormat.YOLO_DARKNET:
            return YoloDarknetReader(self.path_to_folder_with_images,
                                     self.path_to_folder_with_labels)

        if dataset_format == DatasetFormat.CLASSIFICATION:
            return ClassificationReader(self.path_to_folder_with_images)

        raise NotImplementedError()

    def __iter__(self):
        '''Итерация по элементам датасета по генератору
        При итерации упаковываю в класс и возвращаю класс'''
        for elem in self.dataset_reader:
            yield elem
