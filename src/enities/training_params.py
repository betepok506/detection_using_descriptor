from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    output_metric_path: str
    output_transformer_path: str
    output_model_path: str
    use_transformer: bool = field(default=False)
    model_type: str = field(default="SVM")
    random_state: int = field(default=42)