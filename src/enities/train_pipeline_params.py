from dataclasses import dataclass
from src.enities.splitting_params import SplittingParams
from src.enities.training_params import TrainingParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    use_mlflow: bool
    url_mlflow: str
    name_training_in_mlflow: str
    splitting_params: SplittingParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))