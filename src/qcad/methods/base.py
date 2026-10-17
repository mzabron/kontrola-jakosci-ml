from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    scores: np.ndarray
    maps: np.ndarray | None = None

    def __post_init__(self):
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        if self.maps is not None:
            self.maps = np.asarray(self.maps, dtype=np.float32)
            if self.maps.ndim != 3 or len(self.maps) != len(self.scores):
                raise ValueError(f"expected maps of shape (N, H, W), got {self.maps.shape}")


class AnomalyMethod(ABC):
    name: str = "base"
    produces_maps: bool = False

    @abstractmethod
    def fit(self, images: np.ndarray) -> "AnomalyMethod":
        ...

    @abstractmethod
    def predict(self, images: np.ndarray) -> Prediction:
        ...

    def score(self, images: np.ndarray) -> np.ndarray:
        return self.predict(images).scores

    def get_config(self) -> dict:
        return {}
