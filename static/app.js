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
  lastSentLeft: 0,
  lastSentRight: 0,
  isMoving: false,
  lastGazeX: window.innerWidth / 2,
  lastGazeY: window.innerHeight / 2
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
  btnEnterSystem: document.getElementById("btn-enter-system"),
  captureCanvas: document.getElementById("capture-canvas"),

  mainContainer: document.getElementById("main-container"),
  openmvFeed: document.getElementById("openmv-feed"),
  feedFallback: document.getElementById("feed-fallback"),
  focusReticle: document.getElementById("focus-reticle"),
  gazeCursor: document.getElementById("gaze-cursor"),
  leftTriggerZone: document.getElementById("left-trigger-zone"),
  rightTriggerZone: document.getElementById("right-trigger-zone"),

  chkShowCursor: document.getElementById("chk-show-cursor"),
  chkSimulateMouse: document.getElementById("chk-simulate-mouse"),

  boardIndicator: document.getElementById("board-indicator"),
  boardStatusText: document.getElementById("board-status-text"),
  logBox: document.getElementById("log-box"),

  loadingOverlay: document.getElementById("loading-overlay"),
  loadingTitle: document.getElementById("loading-title"),
  loadingDesc: document.getElementById("loading-desc"),
};

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
    };

    els.btnEnterSystem.disabled = false;
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

// Capture current webcam frame as Base64 JPEG
function captureFrame() {
  els.captureCanvas.width = 640;
  els.captureCanvas.height = 480;
  const ctx = els.captureCanvas.getContext("2d");
  ctx.drawImage(els.webcamPreview, 0, 0, 640, 480);
  return els.captureCanvas.toDataURL("image/jpeg", 0.85);
}

// Capture frame as Binary Blob
function captureFrameBlob() {
  els.captureCanvas.width = 640;
  els.captureCanvas.height = 480;
  const ctx = els.captureCanvas.getContext("2d");
  ctx.drawImage(els.webcamPreview, 0, 0, 640, 480);
  return new Promise((resolve) => {
    els.captureCanvas.toBlob((blob) => {
      resolve(blob);
    }, "image/jpeg", 0.85);
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
      log(`校準數據點 [${pointIdx}] 收集成功 (${state.collectedCount}/9)`);
    } else {
      log(`數據點 [${pointIdx}] 收集失敗: ${result.error || "臉部未對準"}`);
    }
  } catch (e) {
    log(`發送樣本錯誤: ${e.message}`);
  }

  // Trigger fine-tuning if all 9 points are collected
  if (state.collectedCount >= 9) {
    await runOnlineFineTuning();
  }
}

// Run Online Model Training / Fine-tuning
async function runOnlineFineTuning() {
  els.loadingOverlay.classList.remove("hidden");
  els.loadingTitle.textContent = "開發板連線中，資料同步中...";
  els.loadingDesc.textContent = "請保持直視螢幕，正在為您在線微調個人化眼動神經網路模型...";

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
      log(`神經網路在線微調完成！平均誤差為: ${data.best_val_px_error.toFixed(1)} px`);
      state.calibrated = true;
      state.testing = true;
      
      // Start websocket predictions
      runPredictionLoop(modelName);
    } else {
      log(`微調失敗: ${data.error}`);
      alert("微調失敗: " + data.error);
    }
  } catch (e) {
    log(`連線微調 API 錯誤: ${e.message}`);
  } finally {
    els.loadingOverlay.classList.add("hidden");
  }
}

// Websocket Real-time Gaze Prediction Loop
let activeWS = null;
async function runPredictionLoop(modelName) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/predict/ws`;
  const ws = new WebSocket(wsUrl);
  activeWS = ws;

  let isProcessing = false;

  ws.onopen = () => {
    log("眼動即時預測串流已啟動");
    ws.send(JSON.stringify({ model_name: modelName }));
  };

  ws.onerror = (err) => {
    isProcessing = false;
  };

  ws.onclose = () => {
    log("眼動預測串流已關閉");
    isProcessing = false;
  };

  ws.onmessage = (event) => {
    isProcessing = false;
    if (!state.testing) return;
    try {
      const data = JSON.parse(event.data);
      if (data.ok) {
        const xNorm = data.screen_xy_norm[0];
        const yNorm = data.screen_xy_norm[1];
        
        // Map normalized [-1, 1] coordinates to actual viewport pixels
        const x = ((xNorm + 1.0) * 0.5) * window.innerWidth;
        const y = ((yNorm + 1.0) * 0.5) * window.innerHeight;
        
        let finalX = x;
        let finalY = y;
        
        // Filter out jitter
        const nowTs = performance.now();
        finalX = filterX.filter(x, nowTs);
        finalY = filterY.filter(y, nowTs);

        // Update cursor
        handleGazeInput(finalX, finalY);
      }
    } catch (err) {
      console.error("Gaze data parse error", err);
    }
  };

  while (state.testing && ws.readyState !== WebSocket.CLOSED) {
    if (ws.readyState === WebSocket.OPEN && !isProcessing && !els.chkSimulateMouse.checked) {
      try {
        const blob = await captureFrameBlob();
        if (blob) {
          isProcessing = true;
          ws.send(blob);
        }
      } catch (err) {
        isProcessing = false;
      }
    }
    await new Promise(r => setTimeout(r, state.frameInterval));
  }
}

// Handle Gaze Input Coordinates (from eye tracking or mouse simulator)
let lastPanTime = 0;
let gazeStayTimer = null;
let lastGazeGridX = -1;
let lastGazeGridY = -1;

function handleGazeInput(x, y) {
  state.lastGazeX = x;
  state.lastGazeY = y;

  if (els.chkShowCursor.checked) {
    els.gazeCursor.classList.remove("hidden");
    els.gazeCursor.style.left = `${x}px`;
    els.gazeCursor.style.top = `${y}px`;
  } else {
    els.gazeCursor.classList.add("hidden");
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
      sendPanAngle(state.currentPanAngle);
      lastPanTime = now;
    } else if (xNorm > 0.6) {
      // Look Right
      els.rightTriggerZone.classList.add("active-gaze");
      els.leftTriggerZone.classList.remove("active-gaze");
      state.currentPanAngle = Math.min(state.currentPanAngle + 5, 90);
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
    if (data.ok) {
      log(`鏡頭轉動至: ${angle}°`);
    }
  } catch (e) {
    console.error("Pan request failed", e);
  }
}

// Send Exposure ROI focus command
async function triggerExposureRoi(screenX, screenY) {
  const rect = els.openmvFeed.getBoundingClientRect();
  
  // Check if coordinates land within the OpenMV video element
  if (screenX >= rect.left && screenX <= rect.right && screenY >= rect.top && screenY <= rect.bottom) {
    const relX = (screenX - rect.left) / rect.width;
    const relY = (screenY - rect.top) / rect.height;
    
    // Map to QQVGA (160x120)
    const pxX = Math.round(relX * 160);
    const pxY = Math.round(relY * 120);

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
      if (data.ok) {
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
function updateCarMovement() {
  let left = 0.0;
  let right = 0.0;
  
  const hasShift = state.activeKeys.has("shift");
  const baseSpeed = hasShift ? 0.8 : 0.4;

  const keyW = state.activeKeys.has("w");
  const keyS = state.activeKeys.has("s");
  const keyA = state.activeKeys.has("a");
  const keyD = state.activeKeys.has("d");

  if (keyW) {
    if (keyA) {
      // Curve left
      left = baseSpeed * 0.3;
      right = baseSpeed;
    } else if (keyD) {
      // Curve right
      left = baseSpeed;
      right = baseSpeed * 0.3;
    } else {
      // Forward
      left = baseSpeed;
      right = baseSpeed;
    }
  } else if (keyS) {
    if (keyA) {
      left = -baseSpeed * 0.3;
      right = -baseSpeed;
    } else if (keyD) {
      left = -baseSpeed;
      right = -baseSpeed * 0.3;
    } else {
      // Backward
      left = -baseSpeed;
      right = -baseSpeed;
    }
  } else if (keyA) {
    // Pure spin left
    left = -baseSpeed;
    right = baseSpeed;
  } else if (keyD) {
    // Pure spin right
    left = baseSpeed;
    right = -baseSpeed;
  }

  // Send request only if wheel speeds changed to prevent API flooding
  if (left !== state.lastSentLeft || right !== state.lastSentRight) {
    state.lastSentLeft = left;
    state.lastSentRight = right;
    sendMoveCommand(left, right);
  }
}

async function sendMoveCommand(left, right) {
  try {
    const res = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ left, right })
    });
    const data = await res.json();
    if (data.ok) {
      if (left === 0 && right === 0) {
        log("小車停止。");
      } else {
        log(`小車行駛中 -> 左輪: ${left.toFixed(1)}, 右輪: ${right.toFixed(1)}`);
      }
    }
  } catch (e) {
    console.error("Move request failed", e);
  }
}

// Window Event Listeners for Keyboard
window.addEventListener("keydown", (e) => {
  const key = e.key.toLowerCase();
  if (["w", "a", "s", "d", "shift"].includes(key)) {
    state.activeKeys.add(key);
    updateCarMovement();
  }
});

window.addEventListener("keyup", (e) => {
  const key = e.key.toLowerCase();
  if (["w", "a", "s", "d", "shift"].includes(key)) {
    state.activeKeys.delete(key);
    updateCarMovement();
  }
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
  }
};

// ==========================================================================
// Setup Welcome Interaction (Stealth Calibration Stage 1 & 2)
// ==========================================================================

// Point 1: Close Warm Tip
els.btnCloseTip.addEventListener("click", () => {
  const rect = els.btnCloseTip.getBoundingClientRect();
  collectStealthPoint(0, rect);
  els.tipBox.style.opacity = "0";
  setTimeout(() => els.tipBox.classList.add("hidden"), 300);
});

// Point 2: Enter System
els.btnEnterSystem.addEventListener("click", async () => {
  const rect = els.btnEnterSystem.getBoundingClientRect();
  
  // Submit calibration sample
  await collectStealthPoint(1, rect);

  // Transition to main screen
  els.welcomeScreen.style.opacity = "0";
  setTimeout(() => {
    els.welcomeScreen.classList.add("hidden");
    els.mainContainer.classList.remove("hidden");
    log("已進入小車操控主介面。請點擊畫面上的提示 widget 關閉鈕，同步校準眼部模型。");
  }, 500);
});

// Points 3 - 9: Floating Widgets Close Button events
document.querySelectorAll(".widget-close").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    const pointIdx = parseInt(btn.getAttribute("data-point-idx"));
    const rect = btn.getBoundingClientRect();
    
    // Collect sample
    collectStealthPoint(pointIdx, rect);

    // Hide the widget
    const widget = btn.closest(".floating-widget");
    widget.classList.add("hidden");
  });
});

// Entry Point
window.addEventListener("DOMContentLoaded", () => {
  els.logBox.textContent = "系統初始化完畢。等待視訊設定...";
  initWebcam();
});
