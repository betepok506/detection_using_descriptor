'''
Данный модуль содержит функции для работы с данными
'''
import datetime
import os
import random
import time
from typing import Tuple, NoReturn

import tqdm
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import cv2 as cv
from scipy.cluster.vq import *
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from src.enities.feature_extraction_params import FeatureExtractionParams
from src.utils.logger import LoggerFormating
from src.enities import SplittingParams, LabelParams
from src.data.dataset_format import *
import json

from enum import Enum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(LoggerFormating())
logger.addHandler(handler)
logger.propagate = False


# class CoordinatesType(Enum):
#     """
#     Class representing if the coordinates are relative to the
#     image size or are absolute values.
#     """
#     Relative = "Relative"
#     Absolute = "Absolute"
#
#
# class LabelsFormat():
#     CXYX2Y2 = "CXYX2Y2"


class VectorizationMethod:
    FIRST_K: str = "first_k"
    BOW: str = "bow"


def extract_features(descriptor,
                     params: FeatureExtractionParams):
    '''

    :param descriptor:
    :param params:
    :return:
    '''
    if not os.path.exists(params.input_data_img_path):
        logger.critical("The images data directory was not found!")
        raise FileNotFoundError('Images folder not found!')

    if params.detection_training and not os.path.exists(params.label_params.input_data_label_path):
        logger.critical("The labels data directory was not found!")
        raise FileNotFoundError('Labels folder not found!')

    folder_examples = f"{params.type_descriptor}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_to_examples = os.path.join(params.path_to_examples, folder_examples)
    probability = 0
    if params.num_examples:
        os.makedirs(path_to_examples, exist_ok=True)
        probability = 0.5

    if params.detection_training:
        dataset = ImagesDataset(params.input_data_img_path,
                                params.label_params.input_data_label_path,
                                params.label_params.annotation_format)
    else:
        dataset = ImagesDataset(params.input_data_img_path)

    target = []
    features = []
    # Счетчик сохраняемых примеров изображений, так же служит для их именования
    cnt_examples = 0
    # Номер текущего, обрабатываемого изображения
    num_img = 0

    logger.info("The process of extracting features has begun!")
    start_ts = time.time()
    for cur_item in tqdm.tqdm(dataset):
        num_img += 1
        logger.debug(f"Processing the image {cur_item.img_name}")
        gray_img = cv.cvtColor(cur_item.img, cv.COLOR_BGR2GRAY)
        for bbox in cur_item.bboxes:
            keypoint, descriptors = descriptor.detect_and_compute(gray_img[bbox._y:bbox._y2, bbox._x:bbox._x2].copy())
            if descriptors is None:
                continue

            if cnt_examples < params.num_examples and probability <= random.random():
                cv.imwrite(os.path.join(path_to_examples, f"{cnt_examples}.png"),
                           descriptor.draw_keypoints(cur_item.img[bbox._y:bbox._y2, bbox._x:bbox._x2],
                                                     keypoint[: min(len(keypoint), params.num_features)]))
                cnt_examples += 1

            if params.vectorization_method == VectorizationMethod.FIRST_K:
                vector_data = descriptor.get_embedding(descriptors, params.num_features)
                features.append(vector_data)
            elif params.vectorization_method == VectorizationMethod.BOW:
                features.append(np.float32(descriptors))
            else:
                raise NotImplementedError()

            target.append(bbox.getClassId())

    end_ts = time.time()
    logger.info("The feature extraction process is over")

    all_time = end_ts - start_ts
    mean_processing_time = all_time / num_img
    logger.info(f"All time: {all_time}")
    logger.info(f"Number of processed images: {num_img}")
    logger.info(f"Average image processing time: {mean_processing_time}")

    if params.vectorization_method == VectorizationMethod.BOW:
        print("Start k-means: %d words" % (params.num_features))
        features_data = features[0][1]
        for descriptor in features[1:]:
            features_data = np.vstack((features_data, descriptor))

        voc, variance = kmeans(features_data, params.num_features, iter=1)
        features = bow(features,
                       voc,
                       num_words=params.num_features)
        logger.info(f"Saving cluster centers at {params.path_to_cluster_centers}")
        os.makedirs(os.path.split(params.path_to_cluster_centers)[0], exist_ok=True)
        cluster_centers = {
            "cluster_centers": voc.tolist()
        }
        with open(params.path_to_cluster_centers, 'w') as file:
            json.dump(cluster_centers, file)


    logger.info("Saving features...")
    pd_data = pd.DataFrame(features)
    pd_data['target'] = target

    os.makedirs(os.path.split(params.output_features_path)[0], exist_ok=True)
    pd_data.to_csv(params.output_features_path, index=False, header=False)

    feature_extraction_metrics = {
        "all_time": all_time,
        "number_images": num_img,
        "average_time": mean_processing_time
    }
    os.makedirs(os.path.split(params.path_to_processing_metrics)[0], exist_ok=True)
    with open(params.path_to_processing_metrics, 'w') as file:
        json.dump(feature_extraction_metrics, file)


def bow(pd: np.array,
        cluster_centers,
        num_words: int):

    cnt_images = len(pd)
    im_features = np.zeros((cnt_images, num_words), "float32")
    for i in range(cnt_images):
        words, distance = vq(pd[i], cluster_centers)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * cnt_images + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # L2 normalization
    im_features = im_features * idf
    im_features = preprocessing.normalize(im_features, norm='l2')
    return im_features


def read_data(data_path: str) -> pd.DataFrame:
    '''
    Функция для чтения csv файла

    Parameter
    ----------

    data_path: `str`
        Путь до csv файла

    Returns
    ----------
    `pd.DataFrame`
        Прочитанный датафрейм
    '''
    return pd.read_csv(data_path, header=None)


def split_train_val_data(X: pd.DataFrame,
                         y: pd.Series,
                         params: SplittingParams) -> Tuple[pd.DataFrame,
                                                           pd.DataFrame,
                                                           pd.Series,
                                                           pd.Series]:
    '''
    Функция для разделения датафрейма на обучающую и валидационную выборку

    Parameter
    -----------
    X: `pd.DataFrame`
        Фрейм с данными для разделения
    params: `SplittingParams`
        Параметры разделения

    Returns
    -----------
    `Tuple[pd.DataFrame, pd.DataFrame]`
        Обучающий, тренировочный наборы данных
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=params.val_size,
                                                        random_state=params.random_state)
    return X_train, X_test, y_train, y_test
