import sys
import os
import time
import uuid
import json
import base64
import threading
from pathlib import Path
from threading import Lock

# Check dependencies and set MOCK_MODE accordingly
MOCK_MODE = False
try:
    import cv2
    import numpy as np
    import torch
    
    # Try importing from the user's downloaded repository
    sys.path.append(r"C:\Users\wenso\Downloads\Personal-Finetuned-Eyetracking\src")
    from unigaze_personalization.preprocess import MediaPipeUniGazePreprocessor
    from unigaze_personalization.transforms import to_unigaze_tensor
    from unigaze_personalization.model import UniGazeFeatureWrapper, load_unigaze_b16
except ImportError as e:
    print(f"\n[Warning] Running in MOCK_MODE: {e}")
    MOCK_MODE = True

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Define directories
WORKSPACE_DIR = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE_DIR / "data" / "sessions"
RUNS_DIR = WORKSPACE_DIR / "runs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OpenMV Eye-Tracking Minimal Server")

# Mount Static Files
static_dir = WORKSPACE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# OpenMV Video Stream Cache Thread
OPENMV_IP = '192.168.50.200'
OPENMV_URL = f'http://{OPENMV_IP}'
latest_frame = bytearray()
frame_lock = Lock()
MJPEG_BOUNDARY = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
MJPEG_END = b'\r\n--frame'

def stream_reader():
    global latest_frame
    from urllib.request import urlopen
    while True:
        try:
            r = urlopen(f'{OPENMV_URL}/video_feed', timeout=5)
            buf = b''
            while True:
                chunk = r.read(8192)
                if not chunk:
                    break
                buf += chunk
                while True:
                    start = buf.find(MJPEG_BOUNDARY)
                    if start < 0:
                        break
                    jpeg_start = start + len(MJPEG_BOUNDARY)
                    end = buf.find(MJPEG_END, jpeg_start)
                    if end < 0:
                        break
                    jpeg = buf[jpeg_start:end]
                    with frame_lock:
                        latest_frame = bytearray(jpeg)
                    buf = buf[end + len(MJPEG_END):]
        except Exception:
            time.sleep(2)

threading.Thread(target=stream_reader, daemon=True).start()

# Setup ML objects or mocks
if not MOCK_MODE:
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
else:
    preprocessor = None
    base_model = None
    model_lock = Lock()
    def get_base_model():
        return None

def get_calib_model(name: str):
    path = RUNS_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _decode_image(data: str) -> np.ndarray:
    if MOCK_MODE:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    return image

def _decode_binary_image(raw: bytes) -> np.ndarray:
    if MOCK_MODE:
        return np.zeros((480, 640, 3), dtype=np.uint8)
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
        connected = len(latest_frame) > 0
    return {"connected": connected, "ip": OPENMV_IP}

@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    def generate():
        while True:
            with frame_lock:
                if len(latest_frame) > 0:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                    yield bytes(latest_frame)
                    yield b'\r\n'
            time.sleep(0.04)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# OpenMV proxies
class MovePayload(BaseModel):
    left: float
    right: float

@app.post("/api/move")
def car_move(payload: MovePayload):
    from urllib.request import urlopen, Request
    try:
        req = Request(
            f"{OPENMV_URL}/api/move",
            data=json.dumps({"left": payload.left, "right": payload.right}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urlopen(req, timeout=3) as res:
            return json.loads(res.read())
    except Exception as e:
        return {"ok": False, "error": str(e)}

class PanPayload(BaseModel):
    angle: float

@app.post("/api/pan")
def camera_pan(payload: PanPayload):
    from urllib.request import urlopen, Request
    try:
        req = Request(
            f"{OPENMV_URL}/api/pan",
            data=json.dumps({"angle": payload.angle}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urlopen(req, timeout=3) as res:
            return json.loads(res.read())
    except Exception as e:
        return {"ok": False, "error": str(e)}

class RoiPayload(BaseModel):
    x: int
    y: int
    w: int
    h: int

@app.post("/api/exposure_roi")
def exposure_roi(payload: RoiPayload):
    from urllib.request import urlopen, Request
    try:
        req = Request(
            f"{OPENMV_URL}/api/exposure_roi",
            data=json.dumps({"x": payload.x, "y": payload.y, "w": payload.w, "h": payload.h}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urlopen(req, timeout=3) as res:
            return json.loads(res.read())
    except Exception as e:
        return {"ok": False, "error": str(e)}

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
    
    # Process with MediaPipe preprocessor
    processed = None
    mp_error = ""
    if not MOCK_MODE:
        try:
            processed = preprocessor.process(image)
        except Exception as e:
            mp_error = str(e)
    else:
        mp_error = "Mock mode bypass"

    manifest_path = session_dir / "manifest.jsonl"
    sample_index = 0
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            sample_index = sum(1 for _ in f)

    stem = f"{sample_index:06d}_calib_{request.point_index:02d}"
    raw_path = session_dir / "raw" / f"{stem}.jpg"
    
    if not MOCK_MODE:
        cv2.imwrite(str(raw_path), image)
    else:
        with open(str(raw_path) + ".txt", "w") as dummy:
            dummy.write("mock raw image data")

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

    if processed is not None and not MOCK_MODE:
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
        
    # Return success in mock mode to allow user interface testing
    return {"ok": True if MOCK_MODE else record["ok"], "sample_index": sample_index, "error": record.get("error", "")}

class TrainRequest(BaseModel):
    data_session_id: str
    base_model_name: str
    output_model_name: str

@app.post("/api/train")
def train_session(request: TrainRequest):
    if MOCK_MODE:
        # Mock successful training
        calibration_data = {
            "W": [[0.0]*2]*6,
            "mean_px_error": 12.5,
            "train_samples": 9
        }
        model_path = RUNS_DIR / f"{request.output_model_name}.json"
        with model_path.open("w", encoding="utf-8") as f:
            json.dump(calibration_data, f, indent=2)
        return {"ok": True, "best_val_px_error": 12.5, "train_samples": 9}

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

@app.websocket("/api/predict/ws")
async def predict_gaze_ws(websocket: WebSocket):
    await websocket.accept()
    model_name = "before"
    model_data = None
    
    if not MOCK_MODE:
        model = get_base_model()
        device = next(model.parameters()).device
    
    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                cfg = json.loads(message["text"])
                model_name = cfg.get("model_name", "before")
                if model_name != "before" and not MOCK_MODE:
                    model_data = get_calib_model(model_name)
                    if model_data is not None:
                        model_data["W"] = np.array(model_data["W"])
            elif "bytes" in message:
                raw_bytes = message["bytes"]
                try:
                    if MOCK_MODE:
                        # Return simulated central gaze prediction
                        await websocket.send_json({
                            "ok": True,
                            "screen_xy_norm": [0.0, 0.0],
                            "gaze_pitch_yaw": [0.0, 0.0]
                        })
                        continue

                    image = _decode_binary_image(raw_bytes)
                    processed = preprocessor.process(image)
                    if processed is None:
                        await websocket.send_json({"ok": False, "error": "No face detected"})
                        continue
                    
                    image_tensor = to_unigaze_tensor(processed.image_rgb).unsqueeze(0).to(device)
                    with torch.no_grad():
                        gaze = model(image_tensor).squeeze(0).cpu().tolist() # [pitch, yaw]
                        
                    if model_name == "before" or model_data is None:
                        # Direct direct linear projection fallback
                        pitch, yaw = gaze[0], gaze[1]
                        pred_x = max(-1.0, min(1.0, yaw * 4.5))
                        pred_y = max(-1.0, min(1.0, pitch * 4.5))
                        pred_xy = [pred_x, pred_y]
                    else:
                        W = model_data["W"]
                        p_curr, y_curr = gaze[0], gaze[1]
                        feat = np.array([y_curr, p_curr, y_curr * y_curr, p_curr * p_curr, y_curr * p_curr, 1.0])
                        pred = feat @ W
                        pred_x = max(-1.0, min(1.0, float(pred[0])))
                        pred_y = max(-1.0, min(1.0, float(pred[1])))
                        pred_xy = [pred_x, pred_y]
                        
                    await websocket.send_json({
                        "ok": True,
                        "screen_xy_norm": pred_xy,
                        "gaze_pitch_yaw": gaze
                    })
                except Exception as e:
                    await websocket.send_json({"ok": False, "error": str(e)})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
