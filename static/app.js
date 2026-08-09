const state = {
  sessionId: "",
  calibrated: false,
  running: false,
  testing: false,
  facingMode: "user",
  frameInterval: 50, // 20 FPS for predictions
  collectedCount: 0,
  targetPoints: {}, // Map point_idx to normalized target coords
  currentPanAngle: 0,
  activeKeys: new Set(),
  lastDriveRequest: "",
  isMoving: false,
  lastGazeX: window.innerWidth / 2,
  lastGazeY: window.innerHeight / 2,
  welcomeTipClosed: false,
  openmvConfigured: false,
  calibrationBusy: false,
  currentCalibrationPoint: null
};

// Adaptive OneEuro Filter for smooth visual cursor movements
class OneEuroFilter {
  constructor(minCutoff = 1.0, beta = 0.007, dcutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dcutoff = dcutoff;
    this.x = null;
    this.dx = null;
    this.lastTime = null;
  }

  alpha(cutoff, rate) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau * rate);
  }

  filter(value, timestamp) {
    if (this.x === null || this.lastTime === null) {
      this.x = value;
      this.dx = 0.0;
      this.lastTime = timestamp;
      return value;
    }

    const dt = (timestamp - this.lastTime) / 1000.0;
    if (dt <= 0) return this.x;

    const rate = 1.0 / dt;
    const dvalue = (value - this.x) * rate;
    const edvalue = this.dx + this.alpha(this.dcutoff, rate) * (dvalue - this.dx);
    this.dx = edvalue;

    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    const filteredValue = this.x + this.alpha(cutoff, rate) * (value - this.x);
    this.x = filteredValue;
    this.lastTime = timestamp;

    return filteredValue;
  }

  reset() {
    this.x = null;
    this.dx = null;
    this.lastTime = null;
  }
}

const filterX = new OneEuroFilter(1.0, 0.007, 1.0);
const filterY = new OneEuroFilter(1.0, 0.007, 1.0);

// DOM Elements
const els = {
  welcomeScreen: document.getElementById("welcome-screen"),
  btnCloseTip: document.getElementById("btn-close-tip"),
  tipBox: document.getElementById("tip-box"),
  webcamPreview: document.getElementById("webcam-preview"),
  cameraSelect: document.getElementById("camera-select"),
  openmvIp: document.getElementById("openmv-ip"),
  btnConnectOpenmv: document.getElementById("btn-connect-openmv"),
  openmvIpStatus: document.getElementById("openmv-ip-status"),
  btnEnterSystem: document.getElementById("btn-enter-system"),
  captureCanvas: document.getElementById("capture-canvas"),

  mainContainer: document.getElementById("main-container"),
  openmvFeed: document.getElementById("openmv-feed"),
  feedFallback: document.getElementById("feed-fallback"),
  focusReticle: document.getElementById("focus-reticle"),
  gazeCursor: document.getElementById("gaze-cursor"),
  leftTriggerZone: document.getElementById("left-trigger-zone"),
  rightTriggerZone: document.getElementById("right-trigger-zone"),
  calibrationLayer: document.getElementById("calibration-layer"),

  chkShowCursor: document.getElementById("chk-show-cursor"),
  chkSimulateMouse: document.getElementById("chk-simulate-mouse"),
  chkEnableDiagonal: document.getElementById("chk-enable-diagonal"),
  chkEnableShift: document.getElementById("chk-enable-shift"),
  keyboardState: document.getElementById("keyboard-state"),
  anglePanel: document.getElementById("angle-panel"),
  angleValue: document.getElementById("angle-value"),
  cameraAngleLine: document.getElementById("camera-angle-line"),

  boardIndicator: document.getElementById("board-indicator"),
  boardStatusText: document.getElementById("board-status-text"),
  logBox: document.getElementById("log-box"),

  btnApiReference: document.getElementById("btn-api-reference"),
  apiReferenceModal: document.getElementById("api-reference-modal"),
  btnCloseApiReference: document.getElementById("btn-close-api-reference"),
  apiReferenceSummary: document.getElementById("api-reference-summary"),
  apiReferenceGrid: document.getElementById("api-reference-grid"),
  apiSpeedExample: document.getElementById("api-speed-example"),

  loadingOverlay: document.getElementById("loading-overlay"),
  loadingTitle: document.getElementById("loading-title"),
  loadingDesc: document.getElementById("loading-desc"),
};

const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;
const captureContext = els.captureCanvas.getContext("2d", {
  alpha: false,
  desynchronized: true
});

const virtualKeys = Array.from(document.querySelectorAll(".virtual-key[data-key]"));
const virtualKeyLabels = {
  w: "W",
  a: "A",
  s: "S",
  d: "D",
  shift: "Shift"
};

const BASIC_DRIVE_APIS = [
  { keys: "W", command: "forward" },
  { keys: "S", command: "backward" },
  { keys: "A", command: "left" },
  { keys: "D", command: "right" }
];
const DIAGONAL_DRIVE_APIS = [
  { keys: "W + A", command: "forward_left" },
  { keys: "W + D", command: "forward_right" },
  { keys: "S + A", command: "backward_left" },
  { keys: "S + D", command: "backward_right" }
];

function isCommandOk(data) {
  return Boolean(data && (data.ok === true || data.status === "ok"));
}

function updateEnterButtonState() {
  els.btnEnterSystem.disabled = !state.welcomeTipClosed ||
    !state.openmvConfigured || !els.cameraSelect.value;
}

function updateVirtualKeyboard() {
  const pressedKeys = [];
  virtualKeys.forEach((keyElement) => {
    const key = keyElement.dataset.key;
    const isPressed = state.activeKeys.has(key);
    keyElement.classList.toggle("is-pressed", isPressed);
    keyElement.setAttribute("aria-pressed", String(isPressed));
    if (isPressed) pressedKeys.push(virtualKeyLabels[key] || key.toUpperCase());
  });

  els.keyboardState.textContent = pressedKeys.length
    ? `目前按下：${pressedKeys.join(" + ")}`
    : "等待鍵盤輸入";
}

function updateAnglePreview() {
  const angle = Math.round(state.currentPanAngle);
  const signedAngle = angle > 0 ? `+${angle}` : `${angle}`;
  els.angleValue.textContent = `${signedAngle}°`;
  els.cameraAngleLine.style.setProperty("--line-angle", `${angle}deg`);
  els.anglePanel?.setAttribute("aria-label", `車體與鏡頭夾角 ${signedAngle} 度`);
}

function updateApiReference() {
  const rows = els.chkEnableDiagonal.checked
    ? BASIC_DRIVE_APIS.concat(DIAGONAL_DRIVE_APIS)
    : BASIC_DRIVE_APIS;
  const fragment = document.createDocumentFragment();

  rows.forEach(({ keys, command }) => {
    const item = document.createElement("div");
    const keyLabel = document.createElement("kbd");
    const apiCode = document.createElement("code");
    item.className = "api-reference-item";
    keyLabel.textContent = keys;
    apiCode.textContent = `GET /api/${command}`;
    item.append(keyLabel, apiCode);
    fragment.appendChild(item);
  });

  els.apiReferenceGrid.replaceChildren(fragment);
  els.apiSpeedExample.classList.toggle("hidden", !els.chkEnableShift.checked);
  const directionCount = els.chkEnableDiagonal.checked ? 8 : 4;
  const speedText = els.chkEnableShift.checked ? "Shift 速度已啟用" : "Shift 速度未啟用";
  els.apiReferenceSummary.textContent = `${directionCount} 方向，${speedText}`;
}

let apiReferenceReturnFocus = null;

function openApiReference() {
  stopCarMovement();
  updateApiReference();
  apiReferenceReturnFocus = document.activeElement;
  els.apiReferenceModal.classList.remove("hidden");
  els.btnCloseApiReference.focus();
}

function closeApiReference() {
  els.apiReferenceModal.classList.add("hidden");
  apiReferenceReturnFocus?.focus();
  apiReferenceReturnFocus = null;
}

els.btnApiReference.addEventListener("click", openApiReference);
els.btnCloseApiReference.addEventListener("click", closeApiReference);
els.apiReferenceModal.addEventListener("click", (event) => {
  if (event.target === els.apiReferenceModal) closeApiReference();
});
els.apiReferenceModal.addEventListener("keydown", (event) => {
  if (event.key === "Tab") {
    event.preventDefault();
    els.btnCloseApiReference.focus();
  }
});

// Logging helper
function log(msg) {
  const time = new Date().toLocaleTimeString();
  els.logBox.textContent = `[${time}] ${msg}\n${els.logBox.textContent}`;
  console.log(`[${time}] ${msg}`);
}

// Check board connection status
async function checkBoardConnection() {
  try {
    const res = await fetch("/api/board_status");
    const data = await res.json();
    if (data.connected) {
      els.boardIndicator.className = "status-indicator online";
      els.boardStatusText.textContent = `已連線 (${data.ip})`;
      els.openmvFeed.classList.remove("hidden");
      els.feedFallback.classList.add("hidden");
    } else {
      els.boardIndicator.className = "status-indicator offline";
      els.boardStatusText.textContent = `離線中 (${data.ip})`;
      els.openmvFeed.classList.add("hidden");
      els.feedFallback.classList.remove("hidden");
    }
  } catch (e) {
    els.boardIndicator.className = "status-indicator offline";
    els.boardStatusText.textContent = "連線失敗";
    els.openmvFeed.classList.add("hidden");
    els.feedFallback.classList.remove("hidden");
  }
}
setInterval(checkBoardConnection, 2000);

// Change the OpenMV car IP from the front end.
async function connectOpenMV() {
  const ip = els.openmvIp.value.trim();
  state.openmvConfigured = false;
  updateEnterButtonState();
  if (!ip) {
    els.openmvIpStatus.textContent = "請先輸入小車 IP";
    return;
  }

  els.btnConnectOpenmv.disabled = true;
  els.openmvIpStatus.textContent = "連線中...";

  try {
    const res = await fetch("/api/openmv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ip })
    });
    const data = await res.json();
    if (!res.ok || !data.ok || data.connected !== true) {
      throw new Error(data.detail || data.error || "Unable to set OpenMV IP");
    }

    els.openmvIp.value = data.ip;
    state.openmvConfigured = true;
    updateEnterButtonState();
    els.openmvIpStatus.textContent = `小車 IP：${data.ip}`;
    log(`小車 IP 已設定為 ${data.ip}`);
    await checkBoardConnection();
  } catch (e) {
    els.openmvIpStatus.textContent = `連線設定失敗：${e.message}`;
    log(`小車連線設定失敗：${e.message}`);
  } finally {
    els.btnConnectOpenmv.disabled = false;
  }
}

els.btnConnectOpenmv.addEventListener("click", connectOpenMV);
els.openmvIp.addEventListener("keydown", (event) => {
  if (event.key === "Enter") connectOpenMV();
});

// Initialize Webcam Selection and Preview
let currentStream = null;

async function initWebcam() {
  try {
    // Request initial permission to get labels
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    stream.getTracks().forEach(t => t.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === "videoinput");
    
    els.cameraSelect.innerHTML = "";
    videoDevices.forEach((dev, idx) => {
      const opt = document.createElement("option");
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `攝影機 ${idx + 1}`;
      els.cameraSelect.appendChild(opt);
    });

    if (videoDevices.length > 0) {
      els.cameraSelect.value = videoDevices[0].deviceId;
      await startWebcam(videoDevices[0].deviceId);
    }
    
    els.cameraSelect.onchange = () => {
      startWebcam(els.cameraSelect.value);
      updateEnterButtonState();
    };

    updateEnterButtonState();
    if (videoDevices.length === 0) {
      log("找不到可用的攝影機");
    }
  } catch (err) {
    alert("無法存取攝影機，請檢查瀏覽器設定！");
    log(`攝影機初始化錯誤: ${err.message}`);
  }
}

async function startWebcam(deviceId) {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
  }
  try {
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: deviceId }, width: 640, height: 480 },
      audio: false
    });
    els.webcamPreview.srcObject = currentStream;
    log("攝影機啟動成功");
  } catch (err) {
    log(`攝影機啟動失敗: ${err.message}`);
  }
}

function drawCaptureFrame() {
  if (!captureContext || !els.webcamPreview.videoWidth || !els.webcamPreview.videoHeight) {
    return false;
  }

  captureContext.drawImage(
    els.webcamPreview,
    0,
    0,
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT
  );
  return true;
}

// Capture current webcam frame as Base64 JPEG.
function captureFrame() {
  if (!drawCaptureFrame()) return "";
  return els.captureCanvas.toDataURL("image/jpeg", 0.78);
}

// Capture frame as Binary Blob without resizing the canvas every frame.
function captureFrameBlob() {
  return new Promise((resolve) => {
    if (!drawCaptureFrame()) {
      resolve(null);
      return;
    }

    els.captureCanvas.toBlob((blob) => {
      resolve(blob);
    }, "image/jpeg", 0.78);
  });
}

// Create eye personalization data session
async function createSession() {
  if (state.sessionId) return;
  try {
    const res = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ participant_id: "stealth_car" })
    });
    const data = await res.json();
    state.sessionId = data.session_id;
    log(`眼動 Session 已建立: ${state.sessionId}`);
  } catch (e) {
    log(`Session 建立失敗: ${e.message}`);
  }
}

// Secretly post a calibration sample
async function collectStealthPoint(pointIdx, rect) {
  await createSession();
  const targetX = rect.left + rect.width / 2;
  const targetY = rect.top + rect.height / 2;
  const targetXNorm = (targetX / window.innerWidth) * 2.0 - 1.0;
  const targetYNorm = (targetY / window.innerHeight) * 2.0 - 1.0;

  const imageData = captureFrame();
  if (!imageData) {
    log("尚未取得可用的 webcam 畫面，略過本次校準資料");
    return false;
  }

  try {
    const res = await fetch("/api/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        image_data: imageData,
        target_x: targetX,
        target_y: targetY,
        target_x_norm: targetXNorm,
        target_y_norm: targetYNorm,
        viewport_width: window.innerWidth,
        viewport_height: window.innerHeight,
        phase: "calibration",
        point_index: pointIdx
      })
    });
    const result = await res.json();
    if (result.ok) {
      state.collectedCount++;
      log(`已收集校準樣本 [${pointIdx}]（${state.collectedCount}/9）。`);
    } else {
      log(`校準樣本 [${pointIdx}] 收集失敗：${result.error || "臉部未對準"}`);
    }
    if (state.collectedCount >= 9) {
      await runOnlineFineTuning();
    }
    return Boolean(result.ok);
  } catch (e) {
    log(`影像樣本上傳失敗：${e.message}`);
    return false;
  }
}

// Run Online Model Training / Fine-tuning
async function runOnlineFineTuning() {
  els.loadingOverlay.classList.remove("hidden");
  els.loadingTitle.textContent = "正在進行個人化微調";
  els.loadingDesc.textContent = "請稍候，系統正在整理剛才收集的影像。";

  try {
    const modelName = `car_model_${state.sessionId}`;
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data_session_id: state.sessionId,
        base_model_name: "0",
        output_model_name: modelName
      })
    });
    const data = await res.json();
    if (data.ok) {
      const errorPx = Number.isFinite(Number(data.best_val_px_error))
        ? `${Number(data.best_val_px_error).toFixed(1)} px`
        : "完成";
      log(`個人化微調完成，驗證誤差：${errorPx}。`);
      state.calibrated = true;
      state.testing = true;
      els.calibrationLayer.classList.add("hidden");
      document.querySelectorAll(".floating-control-panel").forEach((panel) => {
        panel.classList.remove("spotlighted");
      });
      
      // Start websocket predictions
      runPredictionLoop(modelName);
    } else {
      log(`個人化微調失敗：${data.error || "未知錯誤"}`);
      alert("個人化微調失敗：" + (data.error || "未知錯誤"));
    }
  } catch (e) {
    log(`個人化微調 API 失敗：${e.message}`);
  } finally {
    els.loadingOverlay.classList.add("hidden");
  }
}

// Websocket Real-time Gaze Prediction Loop
let activeWS = null;
let kickPredictionLoop = null;
let lastPredictionErrorAt = 0;

function runPredictionLoop(modelName) {
  if (activeWS) {
    try {
      activeWS.close();
    } catch (e) {
      console.warn("Unable to close previous prediction stream", e);
    }
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/predict/ws`;
  const ws = new WebSocket(wsUrl);
  activeWS = ws;

  let isProcessing = false;
  let lastSendAt = 0;
  let responseTimer = null;
  let nextFrameTimer = null;

  const clearPredictionTimers = () => {
    window.clearTimeout(responseTimer);
    window.clearTimeout(nextFrameTimer);
    responseTimer = null;
    nextFrameTimer = null;
  };

  const scheduleNextFrame = () => {
    if (!state.testing || ws.readyState !== WebSocket.OPEN || els.chkSimulateMouse.checked) {
      return;
    }

    window.clearTimeout(nextFrameTimer);
    const elapsed = performance.now() - lastSendAt;
    const delay = Math.max(0, state.frameInterval - elapsed);
    nextFrameTimer = window.setTimeout(sendNextFrame, delay);
  };

  const sendNextFrame = async () => {
    if (!state.testing || ws.readyState !== WebSocket.OPEN || isProcessing || els.chkSimulateMouse.checked) {
      return;
    }

    try {
      const blob = await captureFrameBlob();
      if (!blob || ws.readyState !== WebSocket.OPEN) {
        scheduleNextFrame();
        return;
      }

      isProcessing = true;
      lastSendAt = performance.now();
      ws.send(blob);

      window.clearTimeout(responseTimer);
      responseTimer = window.setTimeout(() => {
        if (!isProcessing) return;
        isProcessing = false;
        console.warn("Gaze prediction response timed out");
        scheduleNextFrame();
      }, Math.max(1000, state.frameInterval * 8));
    } catch (err) {
      isProcessing = false;
      console.error("Gaze frame send failed", err);
      scheduleNextFrame();
    }
  };

  kickPredictionLoop = scheduleNextFrame;

  ws.onopen = () => {
    log("眼動即時預測串流已啟動");
    ws.send(JSON.stringify({ model_name: modelName }));
    scheduleNextFrame();
  };

  ws.onerror = (err) => {
    isProcessing = false;
    clearPredictionTimers();
  };

  ws.onclose = () => {
    log("眼動預測串流已關閉");
    isProcessing = false;
    clearPredictionTimers();
    if (activeWS === ws) {
      activeWS = null;
      kickPredictionLoop = null;
    }
  };

  ws.onmessage = (event) => {
    isProcessing = false;
    window.clearTimeout(responseTimer);
    responseTimer = null;
    if (!state.testing) return;
    try {
      const data = JSON.parse(event.data);
      if (!data.ok) {
        const now = Date.now();
        if (now - lastPredictionErrorAt > 2000) {
          log(`眼動推論暫時無資料：${data.error || "未知錯誤"}`);
          lastPredictionErrorAt = now;
        }
      } else if (Array.isArray(data.screen_xy_norm) && data.screen_xy_norm.length >= 2) {
        const xNorm = Number(data.screen_xy_norm[0]);
        const yNorm = Number(data.screen_xy_norm[1]);
        if (!Number.isFinite(xNorm) || !Number.isFinite(yNorm)) {
          const now = Date.now();
          if (now - lastPredictionErrorAt > 2000) {
            log("眼動推論回傳的座標無效。");
            lastPredictionErrorAt = now;
          }
          scheduleNextFrame();
          return;
        }
        
        // Map normalized [-1, 1] coordinates to actual viewport pixels
        const x = ((xNorm + 1.0) * 0.5) * window.innerWidth;
        const y = ((yNorm + 1.0) * 0.5) * window.innerHeight;
        
        let finalX = x;
        let finalY = y;
        
        // Filter out jitter
        const nowTs = performance.now();
        finalX = filterX.filter(x, nowTs);
        finalY = filterY.filter(y, nowTs);

        queueGazeInput(finalX, finalY);
      } else {
        const now = Date.now();
        if (now - lastPredictionErrorAt > 2000) {
          log("眼動推論回傳格式不完整。");
          lastPredictionErrorAt = now;
        }
      }
    } catch (err) {
      console.error("Gaze data parse error", err);
    }

    scheduleNextFrame();
  };
}

// Handle Gaze Input Coordinates (from eye tracking or mouse simulator)
let lastPanTime = 0;
let gazeStayTimer = null;
let lastGazeGridX = -1;
let lastGazeGridY = -1;
let queuedGazeInput = null;
let gazeRenderPending = false;

function queueGazeInput(x, y) {
  queuedGazeInput = { x, y };
  if (gazeRenderPending) return;

  gazeRenderPending = true;
  window.requestAnimationFrame(() => {
    gazeRenderPending = false;
    if (!queuedGazeInput) return;
    const next = queuedGazeInput;
    queuedGazeInput = null;
    handleGazeInput(next.x, next.y);
  });
}

function isGazeOverControlPanel(x, y) {
  return Array.from(document.querySelectorAll(".floating-control-panel, .calibration-prompt"))
    .some((panel) => {
      if (panel.classList.contains("hidden")) return false;
      const rect = panel.getBoundingClientRect();
      return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
    });
}

function clearGazeActionState() {
  els.leftTriggerZone.classList.remove("active-gaze");
  els.rightTriggerZone.classList.remove("active-gaze");
  clearTimeout(gazeStayTimer);
  gazeStayTimer = null;
  lastGazeGridX = -1;
  lastGazeGridY = -1;
}

function handleGazeInput(x, y) {
  x = Math.max(0, Math.min(window.innerWidth, x));
  y = Math.max(0, Math.min(window.innerHeight, y));
  state.lastGazeX = x;
  state.lastGazeY = y;

  if (els.chkShowCursor.checked) {
    els.gazeCursor.classList.remove("hidden");
    els.gazeCursor.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
  } else {
    els.gazeCursor.classList.add("hidden");
  }

  if (!els.apiReferenceModal.classList.contains("hidden")) {
    clearGazeActionState();
    return;
  }

  // Floating controls are UI targets, not camera-pan zones. Looking at the
  // settings, keyboard, status, or angle panel must not rotate the camera.
  if (isGazeOverControlPanel(x, y)) {
    clearGazeActionState();
    return;
  }

  // Calculate normalized screen X
  const xNorm = (x / window.innerWidth) * 2.0 - 1.0;

  // Camera Pan control by left/right gaze zones
  const now = Date.now();
  if (now - lastPanTime > 300) { // Limit commands to 300ms intervals
    if (xNorm < -0.6) {
      // Look Left
      els.leftTriggerZone.classList.add("active-gaze");
      els.rightTriggerZone.classList.remove("active-gaze");
      state.currentPanAngle = Math.max(state.currentPanAngle - 5, -90);
      updateAnglePreview();
      sendPanAngle(state.currentPanAngle);
      lastPanTime = now;
    } else if (xNorm > 0.6) {
      // Look Right
      els.rightTriggerZone.classList.add("active-gaze");
      els.leftTriggerZone.classList.remove("active-gaze");
      state.currentPanAngle = Math.min(state.currentPanAngle + 5, 90);
      updateAnglePreview();
      sendPanAngle(state.currentPanAngle);
      lastPanTime = now;
    } else {
      els.leftTriggerZone.classList.remove("active-gaze");
      els.rightTriggerZone.classList.remove("active-gaze");
    }
  }

  // Gaze static exposure focus check (3 seconds stay in one region)
  const gridX = Math.floor(x / 50); // 50px cell grid
  const gridY = Math.floor(y / 50);
  
  if (gridX === lastGazeGridX && gridY === lastGazeGridY) {
    if (!gazeStayTimer) {
      gazeStayTimer = setTimeout(() => {
        triggerExposureRoi(x, y);
      }, 3000);
    }
  } else {
    clearTimeout(gazeStayTimer);
    gazeStayTimer = null;
    lastGazeGridX = gridX;
    lastGazeGridY = gridY;
  }
}

// Send camera pan angle command
async function sendPanAngle(angle) {
  try {
    const res = await fetch("/api/pan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ angle })
    });
    const data = await res.json();
    if (isCommandOk(data)) {
      log(`鏡頭轉動至: ${angle}°`);
    }
  } catch (e) {
    console.error("Pan request failed", e);
  }
}

// Send Exposure ROI focus command
async function triggerExposureRoi(screenX, screenY) {
  const box = els.openmvFeed.getBoundingClientRect();
  const sourceWidth = els.openmvFeed.naturalWidth || 160;
  const sourceHeight = els.openmvFeed.naturalHeight || 120;
  const sourceAspect = sourceWidth / sourceHeight;
  const boxAspect = box.width / box.height;

  let contentWidth = box.width;
  let contentHeight = box.height;
  if (boxAspect > sourceAspect) {
    contentWidth = box.height * sourceAspect;
  } else {
    contentHeight = box.width / sourceAspect;
  }

  const contentLeft = box.left + (box.width - contentWidth) / 2;
  const contentTop = box.top + (box.height - contentHeight) / 2;

  // Check the actual image content, excluding object-fit letterboxing.
  if (screenX >= contentLeft && screenX <= contentLeft + contentWidth &&
      screenY >= contentTop && screenY <= contentTop + contentHeight) {
    const relX = (screenX - contentLeft) / contentWidth;
    const relY = (screenY - contentTop) / contentHeight;

    // Map to QQVGA (160x120).
    const pxX = Math.max(0, Math.min(159, Math.round(relX * 160)));
    const pxY = Math.max(0, Math.min(119, Math.round(relY * 120)));

    // Show visual reticle animation
    els.focusReticle.style.left = `${screenX}px`;
    els.focusReticle.style.top = `${screenY}px`;
    els.focusReticle.classList.remove("hidden");
    
    log(`眼動注視曝光對焦中: (${pxX}, ${pxY})`);

    try {
      const res = await fetch("/api/exposure_roi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          x: pxX - 10,
          y: pxY - 10,
          w: 20,
          h: 20
        })
      });
      const data = await res.json();
      if (isCommandOk(data)) {
        log(`曝光區域設定成功: (${pxX}, ${pxY})`);
      }
    } catch (e) {
      console.error("ROI focus request failed", e);
    }

    setTimeout(() => {
      els.focusReticle.classList.add("hidden");
    }, 2000);
  }
}

// ==========================================================================
// Keyboard Controls (WASD + Shift)
// ==========================================================================
function resolveDriveCommand() {
  const keyW = state.activeKeys.has("w");
  const keyS = state.activeKeys.has("s");
  const keyA = state.activeKeys.has("a");
  const keyD = state.activeKeys.has("d");
  const diagonalEnabled = els.chkEnableDiagonal.checked;

  if (keyW) {
    if (diagonalEnabled && keyA) return "forward_left";
    if (diagonalEnabled && keyD) return "forward_right";
    return "forward";
  }
  if (keyS) {
    if (diagonalEnabled && keyA) return "backward_left";
    if (diagonalEnabled && keyD) return "backward_right";
    return "backward";
  }
  if (keyA) return "left";
  if (keyD) return "right";
  return "stop";
}

function buildDriveRequest(command) {
  const useShiftSpeed = command !== "stop" &&
    els.chkEnableShift.checked && state.activeKeys.has("shift");
  return `/api/${command}${useShiftSpeed ? "?speed=true" : ""}`;
}

function updateCarMovement() {
  const command = resolveDriveCommand();
  const path = buildDriveRequest(command);
  if (path === state.lastDriveRequest) return;

  state.lastDriveRequest = path;
  sendDriveCommand(path, command);
}

async function sendDriveCommand(path, command) {
  try {
    const res = await fetch(path);
    const data = await res.json();
    if (isCommandOk(data)) {
      log(command === "stop" ? "小車停止。" : `小車指令：${path}`);
    } else {
      log(`API 尚未實作或執行失敗：${path}`);
    }
  } catch (e) {
    console.error(`Drive request failed: ${path}`, e);
  }
}

function refreshDriveOptions() {
  state.lastDriveRequest = "";
  updateApiReference();
  updateCarMovement();
}

els.chkEnableDiagonal.addEventListener("change", refreshDriveOptions);
els.chkEnableShift.addEventListener("change", refreshDriveOptions);

// Window Event Listeners for Keyboard
window.addEventListener("keydown", (e) => {
  if (els.mainContainer.classList.contains("hidden")) return;
  if (!els.apiReferenceModal.classList.contains("hidden")) {
    if (e.key === "Escape") closeApiReference();
    return;
  }
  const key = e.key.toLowerCase();
  if (["w", "a", "s", "d", "shift"].includes(key)) {
    state.activeKeys.add(key);
    updateVirtualKeyboard();
    updateCarMovement();
    e.preventDefault();
  }
});

window.addEventListener("keyup", (e) => {
  const key = e.key.toLowerCase();
  if (["w", "a", "s", "d", "shift"].includes(key)) {
    state.activeKeys.delete(key);
    updateVirtualKeyboard();
    updateCarMovement();
    e.preventDefault();
  }
});

function stopCarMovement() {
  state.activeKeys.clear();
  updateVirtualKeyboard();
  const stopPath = "/api/stop";
  if (state.lastDriveRequest === stopPath) return;
  state.lastDriveRequest = stopPath;
  sendDriveCommand(stopPath, "stop");
}

window.addEventListener("blur", stopCarMovement);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopCarMovement();
});

// Mouse Simulator Mode
window.addEventListener("mousemove", (e) => {
  if (els.chkSimulateMouse.checked) {
    handleGazeInput(e.clientX, e.clientY);
  }
});

// Toggle simulation
els.chkSimulateMouse.onchange = () => {
  if (els.chkSimulateMouse.checked) {
    log("啟動滑鼠模擬眼動模式，移動滑鼠即可控制鏡頭！");
  } else {
    els.gazeCursor.classList.add("hidden");
    if (kickPredictionLoop) kickPredictionLoop();
  }
};

// ==========================================================================
// Setup Welcome Interaction (Stealth Calibration Stage 1 & 2)
// ==========================================================================

const calibrationPromptIds = {
  2: "widget-center",
  3: "widget-top-left",
  4: "widget-top-right",
  5: "widget-bottom-left",
  6: "widget-bottom-right",
  7: "widget-mid-left",
  8: "widget-mid-right"
};

const calibrationSpotlightIds = {
  3: "visual-settings-panel",
  4: "keyboard-panel",
  5: "board-status-panel",
  6: "angle-panel"
};

function showCalibrationPrompt(pointIdx) {
  const promptId = calibrationPromptIds[pointIdx];
  if (!promptId) return;

  document.querySelectorAll(".calibration-prompt").forEach((prompt) => {
    prompt.classList.add("hidden");
  });
  document.querySelectorAll(".floating-control-panel").forEach((panel) => {
    panel.classList.remove("spotlighted");
  });

  const prompt = document.getElementById(promptId);
  const button = prompt?.querySelector(".acknowledge-button");
  if (!prompt || !button) return;

  button.disabled = false;
  prompt.classList.remove("hidden");
  els.calibrationLayer.classList.remove("hidden");
  const spotlightId = calibrationSpotlightIds[pointIdx];
  if (spotlightId) {
    document.getElementById(spotlightId)?.classList.add("spotlighted");
  }
  state.currentCalibrationPoint = pointIdx;
}

// Point 1: acknowledge the welcome tip and capture immediately.
els.btnCloseTip.addEventListener("click", async () => {
  if (state.calibrationBusy) return;
  state.calibrationBusy = true;
  els.btnCloseTip.disabled = true;

  const rect = els.btnCloseTip.getBoundingClientRect();
  await collectStealthPoint(0, rect);
  state.welcomeTipClosed = true;
  updateEnterButtonState();
  els.tipBox.style.opacity = "0";
  setTimeout(() => els.tipBox.classList.add("hidden"), 220);

  state.calibrationBusy = false;
});

// Point 2: enter the console and reveal exactly one prompt at a time.
els.btnEnterSystem.addEventListener("click", async () => {
  if (state.calibrationBusy || !state.welcomeTipClosed) return;
  state.calibrationBusy = true;
  els.btnEnterSystem.disabled = true;

  const rect = els.btnEnterSystem.getBoundingClientRect();
  await collectStealthPoint(1, rect);

  els.welcomeScreen.style.opacity = "0";
  setTimeout(() => {
    els.welcomeScreen.classList.add("hidden");
    els.mainContainer.classList.remove("hidden");
    showCalibrationPrompt(2);
    log("已進入主介面。提示會依序出現，按下「我知道了」時立即收集影像樣本。");
  }, 220);

  state.calibrationBusy = false;
});

// Points 3 - 9: capture the current button region before revealing the next prompt.
document.querySelectorAll(".acknowledge-button[data-point-idx]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    if (state.calibrationBusy) return;
    const pointIdx = Number.parseInt(btn.getAttribute("data-point-idx"), 10);
    const widget = btn.closest(".calibration-prompt");
    if (!widget || !Number.isFinite(pointIdx)) return;

    state.calibrationBusy = true;
    btn.disabled = true;
    const rect = btn.getBoundingClientRect();
    const collected = await collectStealthPoint(pointIdx, rect);
    if (!collected) {
      btn.disabled = false;
      state.calibrationBusy = false;
      return;
    }
    widget.classList.add("hidden");

    if (pointIdx < 8) {
      showCalibrationPrompt(pointIdx + 1);
    } else if (!state.calibrated) {
      // The final sample starts training in collectStealthPoint. Keep the
      // scrim in place until the loading overlay reports completion.
      els.calibrationLayer.classList.remove("hidden");
    }

    state.calibrationBusy = false;
  });
});

// Entry Point
window.addEventListener("DOMContentLoaded", () => {
  els.logBox.textContent = "系統初始化完畢。等待視訊設定...";
  updateVirtualKeyboard();
  updateAnglePreview();
  updateApiReference();
  initWebcam();
});
