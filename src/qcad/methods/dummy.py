from __future__ import annotations

import numpy as np

from .base import AnomalyMethod, Prediction


class RandomMethod(AnomalyMethod):
    name = "random"
    produces_maps = True

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def fit(self, images: np.ndarray) -> "RandomMethod":
        self._rng = np.random.default_rng(self.seed)
        return self

    def predict(self, images: np.ndarray) -> Prediction:
        n, h, w = images.shape[:3]
        return Prediction(scores=self._rng.random(n),
                          maps=self._rng.random((n, h, w), dtype=np.float32))

    def model_size(self) -> int:
        return 0

    def get_config(self) -> dict:
        return {"seed": self.seed}
