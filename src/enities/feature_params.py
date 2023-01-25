from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    target_col: Optional[str]
