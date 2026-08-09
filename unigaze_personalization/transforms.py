from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGE_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def to_unigaze_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("empty image")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected RGB image with 3 channels, got {image_rgb.shape}")
    resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def normalized_to_pixels(xy: torch.Tensor, viewport: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            (xy[:, 0] + 1.0) * 0.5 * viewport[:, 0],
            (xy[:, 1] + 1.0) * 0.5 * viewport[:, 1],
        ],
        dim=1,
    )

