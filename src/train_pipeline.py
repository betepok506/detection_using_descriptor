import os
from typing import NoReturn
import hydra
import mlflow
import json
from src.models.model_fit_predict import (
    serialize_model,
    train_model,
    predict_model,
    evaluate_model,
    create_inference_pipeline,
    load_model)

from src.data.make_dataset import split_train_val_data
from src.features.build_features import (
    extract_target,
    create_transformer,
    make_features
)
from src.enities.train_pipeline_params import TrainingPipelineParams
from src.data.make_dataset import read_data
from src.utils.logger import get_logger
import logging
from src.utils.logger import LoggerFormating

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(LoggerFormating())
logger.addHandler(handler)
logger.propagate = False


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def train_pipeline(params: TrainingPipelineParams) -> NoReturn:
    logger.info("Reading data")
    data = read_data(params.input_data_path)
    X, y = extract_target(data, params.feature_params.target_col)
    X_train, X_val, y_train, y_val = split_train_val_data(X, y, params.splitting_params)

    logger.info(f'Validation size: {len(X_val)}')
    logger.info(f'Train size: {len(X_train)}')
    logger.info(f"Use transform: {params.train_params.use_transformer}")
    if params.train_params.use_transformer:
        logger.info('Create transformer')
        transformer = create_transformer()
        logger.info('Transformer training')
        transformer.fit(X_train)
        logger.info(f'Save transformer {params.train_params.output_transformer_path}')
        serialize_model(transformer, params.train_params.output_transformer_path)

        logger.info('Start transformer')
        X_train = make_features(transformer, X_train)

    logger.info(f'Start training {params.train_params.model_type}...')
    model = train_model(X_train, y_train, params.train_params)
    logger.info('End training')

    if params.train_params.use_transformer:
        inference_pipeline = create_inference_pipeline(model, transformer)
    else:
        inference_pipeline = model

    val_pred = predict_model(inference_pipeline, X_val)

    logger.info('Evaluate models')
    metrics = evaluate_model(val_pred, y_val)

    if params.use_mlflow:
        mlflow.set_tracking_uri(params.url_mlflow)
        with mlflow.start_run(run_name=params.name_training_in_mlflow):
            logger.info('Saved metrics to mlflow')
            for metric in metrics:
                mlflow.log_metric(metric, metrics[metric])
            # Register the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=params.name_training_in_mlflow,
                registered_model_name=f"{params.name_training_in_mlflow}_{params.train_params.model_type}")

    logger.info("Save metrics")
    os.makedirs(os.path.split(params.train_params.output_metric_path)[0], exist_ok=True)
    with open(params.train_params.output_metric_path, 'w') as file:
        json.dump(metrics, file)

    logger.info(f"Save model to {params.train_params.output_model_path}")
    serialize_model(model, params.train_params.output_model_path)


if __name__ == "__main__":
    train_pipeline()
