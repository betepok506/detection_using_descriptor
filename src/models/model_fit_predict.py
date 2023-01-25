import pickle
import os
import numpy as np
import cv2 as cv
from typing import Dict, Union
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, recall_score
from src.enities.training_params import TrainingParams
from sklearn.pipeline import Pipeline
import pandas as pd
from imutils.object_detection import non_max_suppression
from src.descriptions.descriptions import DescriptionsTypes, ORB, SIFT, SURF

SklearnClassificationModel = Union[svm.SVC]


class ModelTypes:
    SVM = "SVM"


def train_model(features: pd.DataFrame,
                target: pd.Series,
                params: TrainingParams) -> SklearnClassificationModel:
    if params.model_type == ModelTypes.SVM:
        model = svm.SVC()
    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model


def create_inference_pipeline(model: SklearnClassificationModel,
                              transformer: Pipeline) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def predict_model(model: Pipeline or SklearnClassificationModel,
                  features: pd.DataFrame) -> np.array:
    return model.predict(features)


def evaluate_model(y_pred: np.array, y_true: np.array) -> Dict[str, float]:
    return {
        "r2_score": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, average="micro")
    }


def load_model(file_path: str) -> object:
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model


def serialize_model(model: object, output_path: str) -> str:
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    return output_path


class BaseDescriptor():
    def __init__(self):
        self.transformer = None
        self.descriptor = None


class ClassificationDescriptor():
    def __init__(self,
                 path_to_model: str,
                 path_to_transformer: str,
                 descriptor_type: str,
                 cluster_center: np.array):
        transformer = self._load_model(path_to_transformer)
        model = self._load_model(path_to_model)
        self.pipelane = self._create_inference_pipeline(model, transformer)
        self.cluster_center = cluster_center

        if descriptor_type == DescriptionsTypes.ORB:
            self.descriptor = ORB()
        elif descriptor_type == DescriptionsTypes.SIFT:
            self.descriptor = SIFT()
        else:
            raise NotImplementedError()

    def predict(self, img: np.array, embedding_size: int = 20):
        if len(img.shape) != 2:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        keypoint, descriptors = self.descriptor.detect_and_compute(img)
        if descriptors is None:
            return None, None

        if self.cluster_center is None:
            vector = self.descriptor.get_embedding(descriptors, embedding_size)
            new_vector = np.reshape(vector, (1, len(vector)))
        else:
            #TODO Доделать, прокинуть TF-IDf и конфигурацию обучения
            im_features = np.zeros((1, num_words), "float32")
            words, distance = vq(descriptors, self.cluster_centers)
            for w in words:
                im_features[i][w] += 1

        pred = self.pipelane.predict(new_vector)
        p = self.pipelane.decision_function(new_vector)
        p = p.reshape(len(p[0]))
        probability = np.exp(p) / np.sum(np.exp(p), axis=0, keepdims=True)
        return pred[0], probability[np.argmax(probability)]

    def _load_model(self, file_path: str):
        return load_model(file_path)

    def _create_inference_pipeline(self,
                                   model: SklearnClassificationModel,
                                   transformer: ColumnTransformer) -> Pipeline:
        return create_inference_pipeline(model, transformer)


class DetectionDescriptor():
    def __init__(self,
                 path_to_model: str,
                 path_to_transformer: str,
                 descriptor_type: str,
                 cluster_center: np.array or None=None):
        self.classification_descriptor = ClassificationDescriptor(path_to_model,
                                                                  path_to_transformer,
                                                                  descriptor_type,
                                                                  cluster_center)


    def predict(self, img: np.array,
                embedding_size: int = 30,
                pred_threshold: float = 0.5):
        rects = self._selective_search(img)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        detections_bbox = []
        probability_bbox = []
        for rect in rects:
            x, y, w, h = rect
            pred, probability = self.classification_descriptor.predict(gray_img[y:y + h, x:x + w],
                                                                       embedding_size)
            if pred is None:
                continue

            if probability > pred_threshold:
                detections_bbox.append([x, y, x + w, y + h, pred])
                probability_bbox.append(probability)

        detections_bbox = np.array(detections_bbox)
        probability_bbox = np.array(probability_bbox)
        bbox_pick = non_max_suppression(detections_bbox,
                                        probs=probability_bbox,
                                        overlapThresh=0.5)
        return bbox_pick

    def _sliding_window(self, img, step_row: int = 8, step_col: int = 8):
        for row in range(0, img.shape[0], step_row):
            for col in range(0, img.shape[1], step_col):
                pass
        pass

    def _selective_search(self, img):
        ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        return rects
