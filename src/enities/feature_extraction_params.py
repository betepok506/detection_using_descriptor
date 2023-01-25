from dataclasses import dataclass, field
from typing import Optional, List
from marshmallow_dataclass import class_schema
from src.enities.label_params import LabelParams
import yaml


@dataclass()
class FeatureExtractionParams:
    input_data_img_path: str
    detection_training: bool
    label_params: LabelParams
    num_features: int
    vectorization_method: str
    type_descriptor: str
    output_features_path: str
    num_examples: int
    path_to_cluster_centers: str
    path_to_examples: str
    path_to_processing_metrics: str


FeatureExtractionParamsSchema = class_schema(FeatureExtractionParams)


def read_feature_extraction_params(path: str) -> FeatureExtractionParams:
    with open(path, "r") as input_stream:
        schema = FeatureExtractionParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
