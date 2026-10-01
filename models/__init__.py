"""Model construction shared by training and checkpoint evaluation."""

from torch import nn

from .dnn import DNNClassifier


def create_model(model_name: str, model_config: dict) -> nn.Module:
    if model_name == "dnn":
        return DNNClassifier(**model_config)
    raise ValueError(f"Unknown model: {model_name}")
