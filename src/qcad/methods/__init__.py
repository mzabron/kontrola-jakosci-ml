from .base import AnomalyMethod, Prediction
from .dummy import RandomMethod

METHODS: dict[str, type[AnomalyMethod]] = {
    RandomMethod.name: RandomMethod,
}


def create_method(name: str, **params) -> AnomalyMethod:
    if name not in METHODS:
        raise KeyError(f"unknown method: {name}")
    return METHODS[name](**params)


__all__ = ["AnomalyMethod", "Prediction", "METHODS", "create_method"]
