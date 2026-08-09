import os
import time
import uuid
import json
import base64
import asyncio
import threading
from pathlib import Path
from threading import Lock
from typing import Optional

WORKSPACE_DIR = Path(__file__).resolve().parent

# Eye-tracking dependencies are installed from requirements.txt. The local
# unigaze_personalization package keeps this repository self-contained.
import cv2
import numpy as np
import torch
from unigaze_personalization.preprocess import MediaPipeUniGazePreprocessor
from unigaze_personalization.transforms import to_unigaze_tensor
from unigaze_personalization.model import UniGazeFeatureWrapper, load_unigaze_b16

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketState

# Define directories
DATA_DIR = WORKSPACE_DIR / "data" / "sessions"
RUNS_DIR = WORKSPACE_DIR / "runs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OpenMV Eye-Tracking Minimal Server")
shutdown_requested = threading.Event()


class SuppressVideoShutdownCancellation:
    """Treat cancellation of the browser's MJPEG request as normal shutdown."""

    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        try:
            await self.application(scope, receive, send)
        except asyncio.CancelledError:
            if scope.get("type") == "http" and scope.get("path") == "/video_feed":
                return
            raise


app.add_middleware(SuppressVideoShutdownCancellation)

# Mount Static Files
static_dir = WORKSPACE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# OpenMV Video Stream Cache Thread
default_openmv_ip = '192.168.50.200'
OPENMV_IP = default_openmv_ip
OPENMV_URL = f'http://{OPENMV_IP}'
OPENMV_STREAM_PATH = '/stream'
OPENMV_PAN_ANGLE = 0
latest_frame = bytearray()
latest_frame_version = 0
last_frame_at = 0.0
frame_lock = Lock()
MJPEG_BOUNDARY_MARKER = b'--frame'
MJPEG_HEADER_END = b'\r\n\r\n'


class OpenMVConfig(BaseModel):
    ip: str


@app.post("/api/openmv")
def configure_openmv(config: OpenMVConfig):
    global OPENMV_IP, OPENMV_URL, OPENMV_PAN_ANGLE
    global latest_frame, latest_frame_version, last_frame_at
    from urllib.request import urlopen

    ip = config.ip.strip()
    if ip.startswith("http://"):
        ip = ip[7:]
    elif ip.startswith("https://"):
        ip = ip[8:]
    ip = ip.rstrip("/")

    if not ip:
        raise HTTPException(status_code=400, detail="OpenMV IP is required")

    new_url = f"http://{ip}"
    try:
        with urlopen(f"{new_url}{OPENMV_STREAM_PATH}", timeout=5) as response:
            status_code = response.getcode()
            content_type = response.headers.get("Content-Type", "")
            first_byte = response.read(1)
        if status_code != 200:
            raise RuntimeError(f"HTTP status {status_code}")
        if "multipart/x-mixed-replace" not in content_type:
            raise RuntimeError("OpenMV response is not an MJPEG stream")
        if not first_byte:
            raise RuntimeError("OpenMV stream returned no data")
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot connect to OpenMV at {ip}: {error}",
        )

    OPENMV_IP = ip
    OPENMV_URL = new_url
    OPENMV_PAN_ANGLE = 0
    with frame_lock:
        latest_frame = bytearray()
        latest_frame_version += 1
        last_frame_at = 0.0
    return {"ok": True, "ip": OPENMV_IP, "connected": True}

def stream_reader():
    global latest_frame, latest_frame_version, last_frame_at
    from urllib.request import urlopen
    while True:
        target_url = OPENMV_URL
        try:
            r = urlopen(f'{target_url}{OPENMV_STREAM_PATH}', timeout=5)
            buf = b''
            while True:
                if target_url != OPENMV_URL:
                    r.close()
                    break
                chunk = r.read(8192)
                if not chunk:
                    break
                buf += chunk
                while True:
                    start = buf.find(MJPEG_BOUNDARY_MARKER)
                    if start < 0:
                        buf = buf[-len(MJPEG_BOUNDARY_MARKER):]
                        break
                    header_end = buf.find(MJPEG_HEADER_END, start)
                    if header_end < 0:
                        break
                    jpeg_start = header_end + len(MJPEG_HEADER_END)
                    next_boundary = buf.find(MJPEG_BOUNDARY_MARKER, jpeg_start)
                    if next_boundary < 0:
                        break
                    jpeg_end = next_boundary
                    if buf[jpeg_end - 2:jpeg_end] == b'\r\n':
                        jpeg_end -= 2
                    jpeg = buf[jpeg_start:jpeg_end]
                    if not jpeg:
                        buf = buf[next_boundary:]
                        continue
                    with frame_lock:
                        latest_frame = bytearray(jpeg)
                        latest_frame_version += 1
                        last_frame_at = time.monotonic()
                    buf = buf[next_boundary:]
                if len(buf) > 2 * 1024 * 1024:
                    buf = buf[-len(MJPEG_BOUNDARY_MARKER):]
        except Exception:
            time.sleep(2)

threading.Thread(target=stream_reader, daemon=True).start()

# Load MediaPipe once and lazily cache the heavier UniGaze model.
preprocessor = MediaPipeUniGazePreprocessor()
base_model = None
model_lock = Lock()

def get_base_model():
    global base_model
    with model_lock:
        if base_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            base_model = UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()
    return base_model

def get_calib_model(name: str):
    path = RUNS_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("W"), list):
            data["W"] = np.asarray(data["W"], dtype=np.float32)
        for stage in data.get("stages", []):
            if isinstance(stage.get("W"), list):
                stage["W"] = np.asarray(stage["W"], dtype=np.float32)
        return data
    except Exception:
        return None


def _clamp_norm(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _apply_calibration(gaze: list[float], model_name: str, model_data: dict | None) -> list[float]:
    """Apply the reference project's single- or multi-stage calibration model."""
    pitch, yaw = float(gaze[0]), float(gaze[1])
    if model_name == "before":
        return [_clamp_norm(yaw * 4.5), _clamp_norm(pitch * 4.5)]
    if model_data is None:
        raise ValueError(f"model {model_name} not found")

    stages = model_data.get("stages")
    if not stages and "W" in model_data:
        stages = [{
            "W": model_data["W"],
            "poly_degree": model_data.get("poly_degree", 2),
        }]
    if not stages:
        raise ValueError(f"model {model_name} has no calibration weights")

    p_curr, y_curr = pitch, yaw
    for stage in stages:
        degree = int(stage.get("poly_degree", 2))
        weights = np.asarray(stage["W"], dtype=np.float32)
        if degree == 1:
            features = np.array([y_curr, p_curr, 1.0], dtype=np.float32)
        else:
            features = np.array([
                y_curr,
                p_curr,
                y_curr * y_curr,
                p_curr * p_curr,
                y_curr * p_curr,
                1.0,
            ], dtype=np.float32)
        prediction = features @ weights
        if prediction.shape[0] < 2:
            raise ValueError(f"model {model_name} has invalid calibration weights")
        y_curr = float(prediction[0])
        p_curr = float(prediction[1])

    return [_clamp_norm(y_curr), _clamp_norm(p_curr)]

def _decode_image(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    return image

def _decode_binary_image(raw: bytes) -> np.ndarray:
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode binary image")
    return image

# API Routes

@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")

@app.get("/api/board_status")
def board_status():
    with frame_lock:
        connected = bool(latest_frame) and (time.monotonic() - last_frame_at) < 5.0
    return {"connected": connected, "ip": OPENMV_IP}


@app.get("/api/health")
def health_check():
    return {"ok": True}

@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    async def generate():
        last_version = -1
        try:
            while True:
                if shutdown_requested.is_set():
                    return
                with frame_lock:
                    current_version = latest_frame_version
                    frame = bytes(latest_frame) if len(latest_frame) > 0 else None
                if frame is not None and current_version != last_version:
                    last_version = current_version
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                    yield frame
                    yield b'\r\n'
                else:
                    await asyncio.sleep(0.01)
        except (asyncio.CancelledError, GeneratorExit):
            # Browser refreshes and Ctrl+C both cancel this long-lived response.
            return
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# OpenMV proxies
def _proxy_openmv_get(path):
    from urllib.request import urlopen

    with urlopen(f"{OPENMV_URL}{path}", timeout=3) as res:
        return json.loads(res.read())


def _simple_drive_path(left, right):
    deadzone = 0.05
    if abs(left) < deadzone and abs(right) < deadzone:
        return "/api/stop"
    if left > 0 and right > 0:
        return "/api/forward"
    if left < 0 and right < 0:
        return "/api/backward"
    if left < 0 and right > 0:
        return "/api/left"
    if left > 0 and right < 0:
        return "/api/right"
    return "/api/stop"


class MovePayload(BaseModel):
    left: float
    right: float

@app.post("/api/move")
def car_move(payload: MovePayload):
    try:
        path = _simple_drive_path(float(payload.left), float(payload.right))
        return _proxy_openmv_get(path)
    except Exception as e:
        return {"ok": False, "error": str(e)}


DRIVE_API_COMMANDS = {
    "forward",
    "backward",
    "left",
    "right",
    "stop",
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
}


@app.get("/api/{command}")
def car_drive_api(command: str, speed: Optional[bool] = None):
    if command not in DRIVE_API_COMMANDS:
        raise HTTPException(status_code=404, detail="Unknown drive command")

    path = f"/api/{command}"
    if speed is True and command != "stop":
        path += "?speed=true"

    try:
        return _proxy_openmv_get(path)
    except Exception as e:
        return {"ok": False, "error": str(e)}

class PanPayload(BaseModel):
    angle: float

@app.post("/api/pan")
def camera_pan(payload: PanPayload):
    global OPENMV_PAN_ANGLE
    try:
        target_angle = max(-90.0, min(90.0, float(payload.angle)))
        if target_angle > OPENMV_PAN_ANGLE:
            path = "/api/pan_right"
        elif target_angle < OPENMV_PAN_ANGLE:
            path = "/api/pan_left"
        else:
            path = "/api/pan_center"
        result = _proxy_openmv_get(path)
        if result.get("ok"):
            OPENMV_PAN_ANGLE = target_angle
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}

class RoiPayload(BaseModel):
    x: int
    y: int
    w: int
    h: int

@app.post("/api/exposure_roi")
def exposure_roi(payload: RoiPayload):
    return {
        "ok": False,
        "status": "unsupported",
        "error": "day4_wifi.py does not implement exposure ROI",
    }

# Eye Personalization Calibration Endpoints

class SessionRequest(BaseModel):
    participant_id: str = "anonymous"

@app.post("/api/session")
def create_session(request: SessionRequest):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_{request.participant_id}_{uuid.uuid4().hex[:8]}"
    session_dir = DATA_DIR / session_id
    for child in ["raw", "crop", "normalized_face"]:
        (session_dir / child).mkdir(parents=True, exist_ok=True)
    meta = {"session_id": session_id, "participant_id": request.participant_id, "created_at": timestamp}
    (session_dir / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "session_id": session_id}

class SampleRequest(BaseModel):
    session_id: str
    image_data: str
    target_x: float
    target_y: float
    target_x_norm: float
    target_y_norm: float
    viewport_width: float
    viewport_height: float
    point_index: int

@app.post("/api/sample")
def add_sample(request: SampleRequest):
    session_dir = DATA_DIR / request.session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        image = _decode_image(request.image_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    
    # Process with the real MediaPipe/UniGaze preprocessor.
    processed = None
    mp_error = ""
    try:
        processed = preprocessor.process(image)
    except Exception as e:
        mp_error = str(e)

    manifest_path = session_dir / "manifest.jsonl"
    sample_index = 0
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            sample_index = sum(1 for _ in f)

    stem = f"{sample_index:06d}_calib_{request.point_index:02d}"
    raw_path = session_dir / "raw" / f"{stem}.jpg"
    
    cv2.imwrite(str(raw_path), image)

    record = {
        "ok": True,
        "sample_index": sample_index,
        "session_id": request.session_id,
        "target_x": request.target_x,
        "target_y": request.target_y,
        "target_x_norm": request.target_x_norm,
        "target_y_norm": request.target_y_norm,
        "viewport_width": request.viewport_width,
        "viewport_height": request.viewport_height,
        "raw_path": str(raw_path.relative_to(session_dir)),
        "created_at_unix": time.time(),
    }

    if processed is not None:
        crop_path = session_dir / "crop" / f"{stem}.jpg"
        norm_path = session_dir / "normalized_face" / f"{stem}.jpg"
        cv2.imwrite(str(crop_path), processed.crop_bgr)
        cv2.imwrite(str(norm_path), processed.image_bgr)
        record.update({
            "crop_path": str(crop_path.relative_to(session_dir)),
            "normalized_face_path": str(norm_path.relative_to(session_dir)),
            "head_pose_pitch_yaw": processed.head_pose_pitch_yaw.tolist(),
            "face_bbox": processed.face_bbox
        })
    else:
        record["ok"] = False
        record["error"] = mp_error

    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        
    return {"ok": record["ok"], "sample_index": sample_index, "error": record.get("error", "")}

class TrainRequest(BaseModel):
    data_session_id: str
    base_model_name: str
    output_model_name: str

@app.post("/api/train")
def train_session(request: TrainRequest):
    session_dir = DATA_DIR / request.data_session_id
    manifest_path = session_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return {"ok": False, "error": "No calibration data found."}
    
    try:
        # Load records
        records = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_base_model()

        gaze_list = []
        target_list = []
        
        for record in records:
            if not record.get("ok"):
                continue
            norm_path = record.get("normalized_face_path")
            if not norm_path:
                continue
            image_path = session_dir / norm_path
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = to_unigaze_tensor(image_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                gaze_tensor = model(image_tensor)
                gaze_vec = gaze_tensor.squeeze(0).cpu().tolist() # [pitch, yaw]
                
            gaze_list.append(gaze_vec)
            target_list.append([record["target_x_norm"], record["target_y_norm"]])

        N = len(gaze_list)
        if N < 6:
            raise ValueError(f"Not enough valid data points (needed >= 6, got {N})")

        # Ridge Regression Fitting (Degree 2)
        X_raw = np.array(gaze_list)
        pitch = X_raw[:, 0]
        yaw = X_raw[:, 1]
        
        X = np.column_stack([
            yaw,
            pitch,
            yaw * yaw,
            pitch * pitch,
            yaw * pitch,
            np.ones(N)
        ])
        
        Y = np.array(target_list)
        alpha = 1e-4
        W = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
        
        # Calculate training errors
        pred_Y = X @ W
        errors = []
        for i in range(N):
            pred_x_px = (pred_Y[i, 0] + 1.0) * 0.5 * 1920.0
            pred_y_px = (pred_Y[i, 1] + 1.0) * 0.5 * 1080.0
            target_x_px = (Y[i, 0] + 1.0) * 0.5 * 1920.0
            target_y_px = (Y[i, 1] + 1.0) * 0.5 * 1080.0
            errors.append(np.sqrt((pred_x_px - target_x_px)**2 + (pred_y_px - target_y_px)**2))
        
        mean_error = float(np.mean(errors))
        
        calibration_data = {
            "W": W.tolist(),
            "mean_px_error": mean_error,
            "train_samples": N
        }
        
        model_path = RUNS_DIR / f"{request.output_model_name}.json"
        with model_path.open("w", encoding="utf-8") as f:
            json.dump(calibration_data, f, indent=2)
            
        return {"ok": True, "best_val_px_error": mean_error, "train_samples": N}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# WebSocket Real-Time Prediction Gaze Link

def _predict_gaze_frame(raw_bytes, model_name, model_data, model, device):
    started_at = time.perf_counter()
    image = _decode_binary_image(raw_bytes)
    processed = preprocessor.process(image)
    if processed is None:
        return {"ok": False, "error": "No face detected"}

    image_tensor = to_unigaze_tensor(processed.image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        gaze = model(image_tensor).squeeze(0).cpu().tolist()  # [pitch, yaw]
    pred_xy = _apply_calibration(gaze, model_name, model_data)

    return {
        "ok": True,
        "screen_xy_norm": pred_xy,
        "gaze_pitch_yaw": gaze,
        "head_pose_pitch_yaw": processed.head_pose_pitch_yaw.tolist(),
        "face_bbox": processed.face_bbox,
        "inference_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
    }

async def _safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    if websocket.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


@app.websocket("/api/predict/ws")
@app.websocket("/api/predict/ws/")
async def predict_gaze_ws(websocket: WebSocket):
    await websocket.accept()
    model_name = "before"
    model_data = None
    model = get_base_model()
    device = next(model.parameters()).device
    
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") is not None:
                try:
                    cfg = json.loads(message["text"])
                    if not isinstance(cfg, dict):
                        raise ValueError("config must be a JSON object")
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    if not await _safe_send_json(websocket, {"ok": False, "error": f"invalid config: {exc}"}):
                        break
                    continue
                model_name = cfg.get("model_name", "before")
                if model_name != "before":
                    model_data = get_calib_model(model_name)
            elif message.get("bytes") is not None:
                raw_bytes = message["bytes"]
                try:
                    result = await asyncio.to_thread(
                        _predict_gaze_frame,
                        raw_bytes,
                        model_name,
                        model_data,
                        model,
                        device,
                    )
                    if not await _safe_send_json(websocket, result):
                        break
                except Exception as e:
                    if not await _safe_send_json(websocket, {"ok": False, "error": str(e)}):
                        break
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    class EyeTrackingServer(uvicorn.Server):
        def handle_exit(self, sig, frame):
            # Let the MJPEG generator finish before Uvicorn starts its
            # graceful-shutdown wait. This keeps Ctrl+C quick and quiet.
            shutdown_requested.set()
            super().handle_exit(sig, frame)

    config = uvicorn.Config(
        app,
        host=os.environ.get("EYETRACK_HOST", "0.0.0.0"),
        port=int(os.environ.get("EYETRACK_PORT", "8000")),
        timeout_graceful_shutdown=1,
    )
    try:
        EyeTrackingServer(config).run()
    except (asyncio.CancelledError, KeyboardInterrupt):
        # Ctrl+C is the intended way to stop this local server. Uvicorn on
        # Windows may surface the signal as CancelledError first; both are
        # normal exits and should not print a traceback.
        pass
