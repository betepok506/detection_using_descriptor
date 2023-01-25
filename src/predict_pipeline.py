import json
import os
from typing import NoReturn
import click
import sys
import numpy as np
import pybboxes as pbx
import logging
from src.utils.logger import LoggerFormating
from src.utils.utils import read_img
from src.models.model_fit_predict import DetectionDescriptor
from src.evaluation.bounding_boxes import BoundingBoxes
from src.evaluation.bounding_box import BoundingBox
from src.evaluation.evaluator import Evaluator
from src.utils.utils_evaluator import *
import cv2 as cv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(LoggerFormating())
logger.addHandler(handler)
logger.propagate = False


# @click.command()
# @click.option('--path_to_model',
#               default='./models/models/model.pkl',
#               help='The path to model to make prediciion')
# @click.command('--path_to_transformer',
#               default='./models/transformers/transform.pkl',
#               help='The path to the transformer for data transformation')
# @click.option('--path_to_data',
#               default='.\data\\raw\images\PatternNet\\0',
#               help='Path to raw data')
# @click.option('--path_to_prediction',
#               default='./models/predictions/predict.csv',
#               help='Path to save prediction')
def predict_detector(path_to_model: str, path_to_transformer: str,
                     path_to_data: str, path_to_label: str,
                     path_to_cluster_center: str,
                     path_to_prediction: str) -> NoReturn:
    if not os.path.exists(path_to_cluster_center):
        logger.critical(f"Not exists!")

    with open(path_to_cluster_center, "r") as file:
        cluster_center = json.load(file)
        cluster_center = np.float32(cluster_center["cluster_centers"])


    logger.info("Start predicting")
    dd = DetectionDescriptor(path_to_model,
                             path_to_transformer,
                             "ORB")
    allBoundingBoxes = BoundingBoxes()
    for file_name in os.listdir(path_to_data):
        img = read_img(os.path.join(path_to_data, file_name))
        if img is None:
            continue

        pred = dd.predict(img, embedding_size=400)

        for (x1, y1, x2, y2, cls) in pred:
            bb = BoundingBox(
                file_name,
                pred,
                x1,
                y1,
                x2,
                y2,
                CoordinatesType.Absolute,
                bbType=BBType.GroundTruth,
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(img, f"{cls}", (x1 + 3, y1 + 15), 1, 1, (0, 255, 00), 2)

        label_file_name = os.path.splitext(file_name)[0]
        with open(os.path.join(path_to_label, label_file_name + '.txt'), "r") as file:
            for line in file:
                print(line.rstrip())
        cv.imshow('Person Detection', img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    predict_detector('./models/models/model_drinking.pkl',
                     './models/transformers/transform_drinking.pkl',
                     '.\data\\raw\images\Drinking',
                     '.\data\\raw\labels\Drinking',
                     "./models/cluster/drinking_cluster.json",
                     'df')