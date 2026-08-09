from __future__ import annotations

import functools

import torch
from torch import nn


@functools.lru_cache(maxsize=2)
def load_unigaze_b16(device: str = "cpu") -> nn.Module:
    import unigaze

    model = unigaze.load("unigaze_b16_joint", device=device)
    model.eval()
    return model


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


class UniGazeFeatureWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def feature_dim(self) -> int:
        return int(self.model.vit.embed_dim)

    def forward(self, image: torch.Tensor, return_features: bool = False):
        features = self.model.vit.forward_features(image)
        pred_gaze = self.model.gaze_fc(features)
        if return_features:
            return pred_gaze, features
        return pred_gaze




