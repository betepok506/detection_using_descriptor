from dataclasses import dataclass


@dataclass()
class LabelParams:
    input_data_label_path: str
    annotation_format: str
    coordinates_type: str
    labels_format: str
    separator: str
