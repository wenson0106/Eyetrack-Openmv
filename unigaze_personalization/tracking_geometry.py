from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np


VALID_METHODS = {"baseline", "head_proxy", "iris_metric", "metric_raycast"}


@dataclass(frozen=True)
class ScreenGeometry:
    width_mm: float = 531.0
    height_mm: float = 299.0
    camera_to_top_mm: float = 12.0
    plane_z_mm: float = 0.0

    @classmethod
    def from_env(cls) -> "ScreenGeometry":
        return cls(
            width_mm=float(os.environ.get("EYETRACK_SCREEN_WIDTH_MM", "531")),
            height_mm=float(os.environ.get("EYETRACK_SCREEN_HEIGHT_MM", "299")),
            camera_to_top_mm=float(os.environ.get("EYETRACK_CAMERA_TO_SCREEN_TOP_MM", "12")),
            plane_z_mm=float(os.environ.get("EYETRACK_SCREEN_PLANE_Z_MM", "0")),
        )

    def norm_to_camera(self, point_norm: np.ndarray) -> np.ndarray:
        x_norm, y_norm = np.asarray(point_norm, dtype=np.float64).reshape(2)
        return np.array(
            [
                x_norm * self.width_mm * 0.5,
                self.camera_to_top_mm + (y_norm + 1.0) * self.height_mm * 0.5,
                self.plane_z_mm,
            ]
        )

    def camera_to_norm(self, point_camera: np.ndarray) -> np.ndarray:
        x_mm, y_mm = np.asarray(point_camera, dtype=np.float64).reshape(3)[:2]
        return np.array(
            [
                2.0 * x_mm / self.width_mm,
                2.0 * (y_mm - self.camera_to_top_mm) / self.height_mm - 1.0,
            ]
        )


def experiment_config(root: Path) -> dict:
    config_path = root / "experiment.json"
    data = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    method = os.environ.get("EYETRACK_METHOD", data.get("method", "baseline"))
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown EYETRACK_METHOD={method!r}; expected one of {sorted(VALID_METHODS)}")
    return {**data, "method": method}


def pitchyaw_to_vector(gaze_pitch_yaw: np.ndarray) -> np.ndarray:
    pitch, yaw = np.asarray(gaze_pitch_yaw, dtype=np.float64).reshape(2)
    vector = np.array(
        [np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)],
        dtype=np.float64,
    )
    return vector / max(np.linalg.norm(vector), 1e-9)


def denormalize_gaze(gaze_pitch_yaw: np.ndarray, rotation_norm: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_norm, dtype=np.float64).reshape(3, 3).T @ pitchyaw_to_vector(gaze_pitch_yaw)
    vector /= max(np.linalg.norm(vector), 1e-9)
    # UniGaze uses the camera-to-face axis as +Z.  A screen mounted beside the
    # webcam is behind the eye in camera coordinates, so make the ray face -Z.
    if vector[2] > 0:
        vector = -vector
    return vector


def intersect_screen(origin: np.ndarray, direction: np.ndarray, screen: ScreenGeometry) -> np.ndarray:
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    if abs(direction[2]) < 1e-6:
        raise ValueError("gaze ray is parallel to the screen")
    distance = (screen.plane_z_mm - origin[2]) / direction[2]
    if distance <= 0:
        raise ValueError("gaze ray points away from the screen")
    return origin + distance * direction


def compensate_origin(
    calibrated_norm: np.ndarray,
    baseline_origin: np.ndarray,
    current_origin: np.ndarray,
    screen: ScreenGeometry,
) -> np.ndarray:
    """Keep the calibrated visual ray and move only its 3D origin."""
    baseline_origin = np.asarray(baseline_origin, dtype=np.float64).reshape(3)
    current_origin = np.asarray(current_origin, dtype=np.float64).reshape(3)
    target_camera = screen.norm_to_camera(calibrated_norm)
    direction = target_camera - baseline_origin
    hit = intersect_screen(current_origin, direction, screen)
    return screen.camera_to_norm(hit)


def ridge_features(gaze_pitch_yaw: np.ndarray) -> np.ndarray:
    pitch, yaw = np.asarray(gaze_pitch_yaw, dtype=np.float64).reshape(2)
    return np.array([yaw, pitch, yaw * yaw, pitch * pitch, yaw * pitch, 1.0])


def apply_method(
    method: str,
    gaze_pitch_yaw: np.ndarray,
    calibrated_norm: np.ndarray,
    processed,
    model_data: dict | None,
    screen: ScreenGeometry,
) -> tuple[np.ndarray, dict]:
    if method == "baseline" or model_data is None:
        return np.asarray(calibrated_norm, dtype=np.float64), {"method": "baseline"}

    if method in {"head_proxy", "iris_metric"}:
        field = "eye_origin_proxy_cam_mm" if method == "head_proxy" else "eye_origin_iris_cam_mm"
        baseline = model_data.get(f"baseline_{field}")
        if baseline is None:
            raise ValueError(f"calibration model does not contain {field} baseline")
        current = np.asarray(getattr(processed, field), dtype=np.float64)
        corrected = compensate_origin(calibrated_norm, baseline, current, screen)
        return corrected, {
            "method": method,
            "eye_origin_cam_mm": current.tolist(),
            "baseline_eye_origin_cam_mm": baseline,
        }

    if method == "metric_raycast":
        weights = model_data.get("W_raycast")
        if weights is None:
            raise ValueError("calibration model does not contain W_raycast")
        origin = np.asarray(processed.eye_origin_iris_cam_mm, dtype=np.float64)
        direction = denormalize_gaze(gaze_pitch_yaw, processed.rotation_norm)
        hit_norm = screen.camera_to_norm(intersect_screen(origin, direction, screen))
        corrected = np.array([hit_norm[0], hit_norm[1], 1.0]) @ np.asarray(weights, dtype=np.float64)
        return corrected, {
            "method": method,
            "raw_raycast_xy_norm": hit_norm.tolist(),
            "eye_origin_cam_mm": origin.tolist(),
            "gaze_direction_cam": direction.tolist(),
        }

    raise ValueError(f"unsupported tracking method: {method}")
