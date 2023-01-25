from typing import Tuple, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.pipeline import Pipeline
from src.enities.feature_params import FeatureParams


def extract_target(data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        return data.drop(target_column, axis=1), data[target_column]
    except:
        return data.iloc[:, :-1], data.iloc[:, -1]


def drop_columns(data: pd.DataFrame, columns_to_delete: list) -> pd.DataFrame:
    return data.drop(columns_to_delete, axis=1)


def make_features(transformer: Pipeline,
                  df: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(df)


def create_transformer() -> Pipeline:
    return Pipeline([('standardscaler', StandardScaler())])


# def create_transformer(features_params: FeatureParams) -> ColumnTransformer:
#     transformer = ColumnTransformer(
#         [
#             (
#                 'label_pipeline',
#                 create_label_pipeline(),
#                 features_params.target_col,
#             ),
#         ]
#     )
#     return transformer
