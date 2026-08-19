const camera = document.getElementById("camera");
const canvas = document.getElementById("capture");
const context = canvas.getContext("2d", { alpha: false });
const target = document.getElementById("target");
const statusEl = document.getElementById("status");
const cameraSelect = document.getElementById("cameraSelect");
const calibrateButton = document.getElementById("calibrate");
const testButton = document.getElementById("test");
const resultsBody = document.getElementById("results");

const state = {
  experiment: null,
  stream: null,
  sessionId: "",
  modelName: "",
  socket: null,
  socketBusy: false,
  collecting: false,
  activeTarget: null,
  activeCondition: "",
  samples: [],
  summaries: [],
};

const calibrationPoints = [
  [.08, .09], [.50, .09], [.92, .09], [.08, .50], [.50, .50], [.92, .50],
  [.08, .91], [.50, .91], [.92, .91],
];
const testPoints = [
  [.50, .50], [.10, .10], [.90, .10], [.10, .90], [.90, .90],
  [.50, .15], [.50, .85], [.15, .50], [.85, .50],
  [.30, .30], [.70, .30], [.30, .70], [.70, .70],
];
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function setStatus(text) { statusEl.textContent = text; }
function quantile(values, q) {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}
function moveTarget(point) {
  target.style.left = `${point[0] * innerWidth}px`;
  target.style.top = `${point[1] * innerHeight}px`;
  target.style.display = "block";
}
function frameDataUrl() {
  context.drawImage(camera, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", .9);
}
function frameBlob() {
  context.drawImage(camera, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", .86));
}
async function startCamera(deviceId) {
  state.stream?.getTracks().forEach(track => track.stop());
  const constraints = { video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, audio: false };
  if (deviceId) constraints.video.deviceId = { exact: deviceId };
  state.stream = await navigator.mediaDevices.getUserMedia(constraints);
  camera.srcObject = state.stream;
  await camera.play();
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(item => item.kind === "videoinput");
  cameraSelect.replaceChildren(...devices.map((item, index) => {
    const option = document.createElement("option");
    option.value = item.deviceId;
    option.textContent = item.label || `Camera ${index + 1}`;
    option.selected = state.stream.getVideoTracks()[0]?.getSettings().deviceId === item.deviceId;
    return option;
  }));
  setStatus(`攝影機 ${camera.videoWidth}×${camera.videoHeight} 已就緒。校正時請保持頭部固定。`);
}

async function createSession() {
  const participant = document.getElementById("participant").value.trim() || "anonymous";
  const response = await fetch("/api/session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ participant_id: participant }) });
  const data = await response.json();
  if (!data.ok) throw new Error(data.error || "session failed");
  state.sessionId = data.session_id;
  state.modelName = `benchmark_${state.sessionId}`;
}

async function collectCalibrationPoint(point, index) {
  moveTarget(point);
  setStatus(`校正 ${index + 1}/9：只移動眼睛注視綠點，保持頭部固定。`);
  await sleep(1100);
  const x = point[0] * innerWidth;
  const y = point[1] * innerHeight;
  for (let repeat = 0; repeat < 3; repeat += 1) {
    const response = await fetch("/api/sample", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId, image_data: frameDataUrl(), target_x: x, target_y: y,
        target_x_norm: point[0] * 2 - 1, target_y_norm: point[1] * 2 - 1,
        viewport_width: innerWidth, viewport_height: innerHeight, point_index: index,
      }),
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || `point ${index + 1} failed`);
    await sleep(100);
  }
}

async function calibrate() {
  calibrateButton.disabled = true;
  testButton.disabled = true;
  document.body.classList.add("running");
  try {
    if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
      await document.documentElement.requestFullscreen().catch(() => {});
    }
    await createSession();
    for (let i = 0; i < calibrationPoints.length; i += 1) await collectCalibrationPoint(calibrationPoints[i], i);
    target.style.display = "none";
    setStatus("正在擬合個人校正與幾何模型…");
    const response = await fetch("/api/train", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_session_id: state.sessionId, base_model_name: "0", output_model_name: state.modelName }),
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "training failed");
    await connectPredictionSocket();
    setStatus(`校正完成（訓練誤差 ${Number(data.best_val_px_error).toFixed(1)} px）。現在移動到指定頭位再開始測試，請勿重新校正。`);
    testButton.disabled = false;
  } catch (error) {
    setStatus(`校正失敗：${error.message}`);
    calibrateButton.disabled = false;
  } finally {
    target.style.display = "none";
    document.body.classList.remove("running");
  }
}

async function connectPredictionSocket() {
  state.socket?.close();
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  state.socket = new WebSocket(`${protocol}//${location.host}/api/predict/ws`);
  await new Promise((resolve, reject) => {
    state.socket.onopen = () => { state.socket.send(JSON.stringify({ model_name: state.modelName })); resolve(); };
    state.socket.onerror = () => reject(new Error("prediction websocket failed"));
  });
  state.socket.onmessage = event => {
    state.socketBusy = false;
    const data = JSON.parse(event.data);
    if (state.collecting && data.ok && state.activeTarget) recordPrediction(data);
    pumpPrediction();
  };
}

async function pumpPrediction() {
  if (!state.collecting || state.socketBusy || state.socket?.readyState !== WebSocket.OPEN) return;
  const blob = await frameBlob();
  if (!blob || !state.collecting) return;
  state.socketBusy = true;
  state.socket.send(blob);
}

async function waitForSocketIdle(timeoutMs = 3000) {
  const deadline = performance.now() + timeoutMs;
  while (state.socketBusy && performance.now() < deadline) await sleep(20);
}

function recordPrediction(data) {
  const x = (data.screen_xy_norm[0] + 1) * .5 * innerWidth;
  const y = (data.screen_xy_norm[1] + 1) * .5 * innerHeight;
  const tx = state.activeTarget[0] * innerWidth;
  const ty = state.activeTarget[1] * innerHeight;
  const dxPx = x - tx;
  const dyPx = y - ty;
  const screen = state.experiment.screen;
  const errorMm = Math.hypot(dxPx * screen.width_mm / innerWidth, dyPx * screen.height_mm / innerHeight);
  const viewingDistanceMm = Number(document.getElementById("viewingDistance").value) || 600;
  state.samples.push({
    participant: document.getElementById("participant").value.trim(),
    method: state.experiment.method, branch: state.experiment.branch,
    condition: state.activeCondition, target_x_px: tx, target_y_px: ty,
    predicted_x_px: x, predicted_y_px: y, error_px: Math.hypot(dxPx, dyPx),
    error_mm: errorMm, error_deg: Math.atan2(errorMm, viewingDistanceMm) * 180 / Math.PI,
    viewing_distance_mm: viewingDistanceMm,
    inference_ms: data.inference_ms, iris_depth_mm: data.iris_depth_mm,
    eye_origin_proxy_cam_mm: data.eye_origin_proxy_cam_mm,
    eye_origin_iris_cam_mm: data.eye_origin_iris_cam_mm,
    timestamp: new Date().toISOString(),
  });
  document.getElementById("sampleCount").textContent = state.samples.length;
}

async function runCondition() {
  testButton.disabled = true;
  calibrateButton.disabled = true;
  const condition = document.getElementById("condition").value;
  state.activeCondition = condition;
  document.body.classList.add("running");
  const startIndex = state.samples.length;
  try {
    for (let i = 0; i < testPoints.length; i += 1) {
      state.activeTarget = testPoints[i];
      moveTarget(state.activeTarget);
      setStatus(`${condition}：測試點 ${i + 1}/${testPoints.length}，請注視綠點。`);
      await sleep(800);
      state.collecting = true;
      pumpPrediction();
      await sleep(1100);
      state.collecting = false;
      await waitForSocketIdle();
    }
    const conditionSamples = state.samples.slice(startIndex);
    if (!conditionSamples.length) throw new Error("此條件沒有有效預測影格，請檢查臉部偵測與模型輸出");
    const errors = conditionSamples.map(row => row.error_px);
    const degrees = conditionSamples.map(row => row.error_deg);
    const summary = {
      condition, n: errors.length,
      mean_px: errors.reduce((a, b) => a + b, 0) / errors.length, p95_px: quantile(errors, .95),
      mean_deg: degrees.reduce((a, b) => a + b, 0) / degrees.length, p95_deg: quantile(degrees, .95),
    };
    state.summaries = state.summaries.filter(item => item.condition !== condition).concat(summary);
    renderResults(summary);
    setStatus(`${condition} 完成。請回到中央休息，或選下一個頭位繼續；不要重新校正。`);
    document.getElementById("exportJson").disabled = false;
    document.getElementById("exportCsv").disabled = false;
  } finally {
    state.collecting = false;
    state.activeTarget = null;
    target.style.display = "none";
    document.body.classList.remove("running");
    testButton.disabled = false;
  }
}

function renderResults(latest) {
  resultsBody.innerHTML = state.summaries.map(row => `<tr><td>${row.condition}</td><td>${row.n}</td><td>${row.mean_deg.toFixed(2)}°</td><td>${row.p95_deg.toFixed(2)}°</td></tr>`).join("");
  document.getElementById("meanError").textContent = `${latest.mean_deg.toFixed(2)}°`;
  document.getElementById("p95Error").textContent = `${latest.p95_deg.toFixed(2)}°`;
}
function download(name, content, type) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([content], { type })); link.download = name; link.click(); URL.revokeObjectURL(link.href);
}
function exportJson() {
  download(`${state.sessionId}_${state.experiment.method}.json`, JSON.stringify({ experiment: state.experiment, viewport: [innerWidth, innerHeight], summaries: state.summaries, samples: state.samples }, null, 2), "application/json");
}
function exportCsv() {
  const keys = ["participant", "method", "branch", "condition", "target_x_px", "target_y_px", "predicted_x_px", "predicted_y_px", "error_px", "error_mm", "error_deg", "viewing_distance_mm", "inference_ms", "iris_depth_mm", "timestamp"];
  const lines = [keys.join(","), ...state.samples.map(row => keys.map(key => JSON.stringify(row[key] ?? "")).join(","))];
  download(`${state.sessionId}_${state.experiment.method}.csv`, lines.join("\n"), "text/csv");
}

cameraSelect.addEventListener("change", () => startCamera(cameraSelect.value).catch(error => setStatus(error.message)));
calibrateButton.addEventListener("click", calibrate);
testButton.addEventListener("click", runCondition);
document.getElementById("exportJson").addEventListener("click", exportJson);
document.getElementById("exportCsv").addEventListener("click", exportCsv);

(async () => {
  try {
    state.experiment = await (await fetch("/api/experiment")).json();
    document.getElementById("method").textContent = `${state.experiment.branch} · ${state.experiment.method}`;
    await startCamera();
  } catch (error) { setStatus(`初始化失敗：${error.message}`); }
})();
