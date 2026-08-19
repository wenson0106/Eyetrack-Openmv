from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import os
from pathlib import Path
from typing import Any
import urllib.request

import cv2
import numpy as np

MP_TO_DLIB_6 = {
    "right_eye_outer": 33,
    "right_eye_inner": 133,
    "left_eye_inner": 362,
    "left_eye_outer": 263,
    "nose_right": 98,
    "nose_left": 327,
}


@dataclass
class NormalizedFace:
    image_rgb: np.ndarray
    image_bgr: np.ndarray
    crop_bgr: np.ndarray
    landmarks: np.ndarray
    landmarks_crop: np.ndarray
    face_bbox: dict[str, float]
    head_pose_pitch_yaw: np.ndarray
    warp_matrix: np.ndarray
    rotation_norm: np.ndarray
    head_rvec_cam: np.ndarray
    head_tvec_cam: np.ndarray
    eye_midpoint_px: np.ndarray
    eye_origin_proxy_cam_mm: np.ndarray
    eye_origin_iris_cam_mm: np.ndarray
    iris_depth_mm: float
    camera_matrix: np.ndarray


def _load_face_model() -> np.ndarray:
    with resources.files("unigaze_personalization.assets").joinpath("face_model.txt").open(
        "r", encoding="utf-8"
    ) as handle:
        return np.loadtxt(handle)


def _dummy_camera(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    focal_length = width * 4.0
    center = (width / 2.0, height / 2.0)
    camera = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    distortion = np.zeros((1, 5), dtype=np.float64)
    return camera, distortion


def _camera_from_frame(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed full-frame camera model.

    The previous crop-relative camera scaled its focal length with the face crop,
    which mathematically removed most lateral/depth motion.  A calibrated matrix
    can be supplied through EYETRACK_CAMERA_FX/FY/CX/CY; otherwise a 60 degree
    horizontal-FOV approximation is used as a device-level fallback.
    """
    height, width = image.shape[:2]
    hfov = float(os.environ.get("EYETRACK_CAMERA_HFOV_DEG", "60"))
    focal_default = width / (2.0 * np.tan(np.deg2rad(hfov) / 2.0))
    fx = float(os.environ.get("EYETRACK_CAMERA_FX", focal_default))
    fy = float(os.environ.get("EYETRACK_CAMERA_FY", focal_default))
    cx = float(os.environ.get("EYETRACK_CAMERA_CX", width / 2.0))
    cy = float(os.environ.get("EYETRACK_CAMERA_CY", height / 2.0))
    camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    distortion = np.zeros((1, 5), dtype=np.float64)
    return camera, distortion


def _point_at_depth(pixel: np.ndarray, depth_mm: float, camera: np.ndarray) -> np.ndarray:
    u, v = np.asarray(pixel, dtype=np.float64).reshape(2)
    fx, fy = camera[0, 0], camera[1, 1]
    cx, cy = camera[0, 2], camera[1, 2]
    return np.array(
        [(u - cx) * depth_mm / fx, (v - cy) * depth_mm / fy, depth_mm],
        dtype=np.float64,
    )


def _eye_origins(landmarks: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    right_center = landmarks[468]
    left_center = landmarks[473]
    eye_midpoint = (right_center + left_center) * 0.5

    iris_diameters = [
        np.linalg.norm(landmarks[469] - landmarks[471]),
        np.linalg.norm(landmarks[470] - landmarks[472]),
        np.linalg.norm(landmarks[474] - landmarks[476]),
        np.linalg.norm(landmarks[475] - landmarks[477]),
    ]
    # A larger inter-eye feature is less noisy than iris diameter and is useful
    # as a relative-position baseline.  63 mm is a population prior, not a user
    # calibration; only displacement from the calibration median is consumed.
    eye_distance_px = max(float(np.linalg.norm(right_center - left_center)), 1.0)
    proxy_depth = float(camera[0, 0] * 63.0 / eye_distance_px)
    valid_iris_diameters = [value for value in iris_diameters if np.isfinite(value) and value > 0.5]
    iris_depth = (
        float(camera[0, 0] * 11.7 / np.median(valid_iris_diameters))
        if valid_iris_diameters
        else proxy_depth
    )
    proxy_origin = _point_at_depth(eye_midpoint, proxy_depth, camera)
    iris_origin = _point_at_depth(eye_midpoint, iris_depth, camera)
    return eye_midpoint, proxy_origin, iris_origin, iris_depth


def _square_crop(image: np.ndarray, points: np.ndarray, scale: float) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    height, width = image.shape[:2]
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    side = max(x_max - x_min, y_max - y_min) * scale
    side = max(side, 32.0)
    x0 = int(round(center_x - side / 2.0))
    y0 = int(round(center_y - side / 2.0))
    x1 = int(round(center_x + side / 2.0))
    y1 = int(round(center_y + side / 2.0))

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)
    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )
    crop = padded[y0 + pad_top : y1 + pad_top, x0 + pad_left : x1 + pad_left]
    offset = np.array([x0, y0], dtype=np.float32)
    bbox = {
        "x": float(max(0, x0)),
        "y": float(max(0, y0)),
        "w": float(min(width, x1) - max(0, x0)),
        "h": float(min(height, y1) - max(0, y0)),
        "x_norm": float(max(0, x0) / width),
        "y_norm": float(max(0, y0) / height),
        "w_norm": float((min(width, x1) - max(0, x0)) / width),
        "h_norm": float((min(height, y1) - max(0, y0)) / height),
    }
    return crop, bbox, offset


def _estimate_head_pose(
    landmarks_2d: np.ndarray,
    face_model_3d: np.ndarray,
    camera: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ok, rvec, tvec = cv2.solvePnP(
        face_model_3d.reshape(6, 1, 3),
        landmarks_2d.astype(np.float64).reshape(6, 1, 2),
        camera,
        distortion,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        raise ValueError("head pose solvePnP failed")
    ok, rvec, tvec = cv2.solvePnP(
        face_model_3d.reshape(6, 1, 3),
        landmarks_2d.astype(np.float64).reshape(6, 1, 2),
        camera,
        distortion,
        rvec,
        tvec,
        True,
    )
    if not ok:
        raise ValueError("head pose refinement failed")
    return rvec, tvec


def _face_center_by_nose(rotation: np.ndarray, translation: np.ndarray, face_model: np.ndarray) -> np.ndarray:
    face_6 = face_model[[20, 23, 26, 29, 15, 19], :]
    fc = np.dot(rotation, face_6.T) + translation.reshape(3, 1)
    two_eye_center = np.mean(fc[:, 0:4], axis=1).reshape(3, 1)
    nose_center = np.mean(fc[:, 4:6], axis=1).reshape(3, 1)
    return np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape(3, 1)


def _normalize_face(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    face_center: np.ndarray,
    head_rvec: np.ndarray,
    camera: np.ndarray,
    focal_norm: float = 960.0,
    distance_norm: float = 600.0,
    roi_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = face_center.reshape(3, 1)
    head_rotation = cv2.Rodrigues(head_rvec)[0]
    distance = np.linalg.norm(center)
    z_scale = distance_norm / distance
    camera_norm = np.array(
        [[focal_norm, 0, roi_size[0] / 2], [0, focal_norm, roi_size[1] / 2], [0, 0, 1.0]]
    )
    scale = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z_scale]])
    head_x = head_rotation[:, 0]
    forward = (center / distance).reshape(3)
    down = np.cross(forward, head_x)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    rotation_norm = np.c_[right, down, forward].T
    warp = np.dot(np.dot(camera_norm, scale), np.dot(rotation_norm, np.linalg.inv(camera)))
    normalized = cv2.warpPerspective(image_bgr, warp, roi_size)
    head_rotation_norm = np.dot(rotation_norm, head_rotation)
    return normalized, warp, head_rotation_norm


def _head_pose_pitch_yaw(head_rotation_norm: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.arcsin(head_rotation_norm[1, 2]),
            np.arctan2(head_rotation_norm[0, 2], head_rotation_norm[2, 2]),
        ],
        dtype=np.float32,
    )


class MediaPipeUniGazePreprocessor:
    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.core import base_options

        model_override = os.environ.get("EYETRACK_FACE_LANDMARKER_PATH")
        if model_override:
            model_path = Path(model_override).expanduser()
        else:
            bundled_model = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
            cache_model = Path.home() / ".cache" / "eyetrack-openmv" / "face_landmarker.task"
            model_path = bundled_model if bundled_model.exists() else cache_model

        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        base_opts = base_options.BaseOptions(model_asset_path=str(model_path))
        opts = FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
        )
        self._face_mesh = FaceLandmarker.create_from_options(opts)
        self._face_model = _load_face_model()
        self._face_model_6 = self._face_model[[20, 23, 26, 29, 15, 19], :]

    def process(self, image_bgr: np.ndarray) -> NormalizedFace:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("empty image")
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        from mediapipe import ImageFormat, Image
        mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
        result = self._face_mesh.detect(mp_image)
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            raise ValueError("no face detected")

        face_landmarks = result.face_landmarks[0]
        landmarks = np.array(
            [[lm.x * width, lm.y * height] for lm in face_landmarks],
            dtype=np.float32,
        )
        crop, bbox, offset = _square_crop(image_bgr, landmarks, scale=2.0)
        landmarks_crop = landmarks - offset
        selected = np.array(
            [
                landmarks[MP_TO_DLIB_6["right_eye_outer"]],
                landmarks[MP_TO_DLIB_6["right_eye_inner"]],
                landmarks[MP_TO_DLIB_6["left_eye_inner"]],
                landmarks[MP_TO_DLIB_6["left_eye_outer"]],
                landmarks[MP_TO_DLIB_6["nose_right"]],
                landmarks[MP_TO_DLIB_6["nose_left"]],
            ],
            dtype=np.float32,
        )
        camera, distortion = _camera_from_frame(image_bgr)
        head_rvec, head_tvec = _estimate_head_pose(selected, self._face_model_6, camera, distortion)
        head_rotation = cv2.Rodrigues(head_rvec)[0]
        face_center = _face_center_by_nose(head_rotation, head_tvec, self._face_model)
        normalized_bgr, warp, head_rotation_norm = _normalize_face(
            image_bgr,
            landmarks,
            face_center,
            head_rvec,
            camera,
        )
        # Recompute the normalization rotation explicitly; the perspective warp
        # also contains scale/intrinsics and cannot be used to denormalize gaze.
        center = face_center.reshape(3)
        distance = np.linalg.norm(center)
        head_x = head_rotation[:, 0]
        forward = center / distance
        down = np.cross(forward, head_x)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        rotation_norm = np.c_[right, down, forward].T
        eye_midpoint, proxy_origin, iris_origin, iris_depth = _eye_origins(landmarks, camera)
        normalized_rgb = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)
        return NormalizedFace(
            image_rgb=normalized_rgb,
            image_bgr=normalized_bgr,
            crop_bgr=crop,
            landmarks=landmarks,
            landmarks_crop=landmarks_crop,
            face_bbox=bbox,
            head_pose_pitch_yaw=_head_pose_pitch_yaw(head_rotation_norm),
            warp_matrix=warp,
            rotation_norm=rotation_norm.astype(np.float32),
            head_rvec_cam=head_rvec.reshape(3).astype(np.float32),
            head_tvec_cam=head_tvec.reshape(3).astype(np.float32),
            eye_midpoint_px=eye_midpoint.astype(np.float32),
            eye_origin_proxy_cam_mm=proxy_origin.astype(np.float32),
            eye_origin_iris_cam_mm=iris_origin.astype(np.float32),
            iris_depth_mm=iris_depth,
            camera_matrix=camera.astype(np.float32),
        )
