from dataclasses import dataclass


@dataclass()
class SplittingParams:
    val_size: float
    random_state: int