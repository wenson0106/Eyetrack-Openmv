# Head-motion robustness experiments

These branches answer one question: can a single fixed-head, nine-point personal calibration remain accurate when the user translates their head? OpenMV is not part of this experiment.

## Branches

| Branch | Runtime method | Additional personal calibration | Main assumption |
| --- | --- | --- | --- |
| `agent/head-motion-benchmark` | Existing UniGaze + degree-2 Ridge | None beyond existing 9 points | Baseline/control |
| `agent/head-proxy-compensation` | Inter-eye scale + eye midpoint + geometric origin compensation | None | Mean interpupillary distance is 63 mm; errors mostly cancel after subtracting the calibration median |
| `agent/iris-metric-compensation` | MediaPipe iris diameter (11.7 mm prior) + geometric origin compensation | None | Iris landmarks are large enough in the webcam image |
| `agent/metric-raycast` | Iris eye origin + denormalized UniGaze 3D ray + screen-plane intersection + 2D affine residual | Same fixed-head 9 points | Camera/screen geometry is configured correctly |

All branches expose the same test page at `http://127.0.0.1:8000/benchmark` and export raw JSON/CSV. The branch method is reported by `/api/experiment` and included in every exported row.

## Device setup (once per webcam/screen, not per user)

For a fair metric comparison, measure the physical display and camera placement. Set these environment variables before starting the server:

```text
EYETRACK_SCREEN_WIDTH_MM=531
EYETRACK_SCREEN_HEIGHT_MM=299
EYETRACK_CAMERA_TO_SCREEN_TOP_MM=12
EYETRACK_CAMERA_HFOV_DEG=60
```

For the best result, replace the FOV approximation with OpenCV camera calibration values:

```text
EYETRACK_CAMERA_FX=...
EYETRACK_CAMERA_FY=...
EYETRACK_CAMERA_CX=...
EYETRACK_CAMERA_CY=...
```

Use 1280×720 or higher, stable illumination, autofocus disabled if the driver permits it, and the same browser zoom/display scaling for every run.

## Human test protocol

1. Check out one branch and start `eye_server.py`.
2. Open `/benchmark`, enter an anonymous participant code, select the webcam, and measure eye-to-screen distance.
3. Run the 9-point calibration once. Keep the head in the central neutral pose; move only the eyes.
4. Without recalibrating, run these conditions in randomized order:
   - central/neutral;
   - left translation 10 cm;
   - right translation 10 cm;
   - 10 cm closer;
   - 10 cm farther;
   - natural continuous translation inside a 20 × 20 × 20 cm box.
5. Return to the neutral pose for 20 seconds between conditions. Do not let the participant see error numbers until the run ends.
6. Export JSON and CSV. Repeat all four branches with the same participant, device, lighting, target order, and distance.
7. Test at least 12 people; 20+ is preferred. Include glasses, contact lenses, different eye colors, and varied face geometry. Counterbalance branch order to reduce learning/fatigue bias.

## Metrics and gates

Evaluate per participant first, then report the median and bootstrap 95% confidence interval across participants.

| Metric | Competitive target | Failure signal |
| --- | ---: | ---: |
| Neutral mean angular error | ≤ 1.0° | > 1.5° |
| Translated-head mean angular error | ≤ 1.5° | > 2.5° |
| Translated-head P95 angular error | ≤ 2.5° | > 4.0° |
| Accuracy loss vs neutral | ≤ 0.5° | > 1.0° |
| Valid-frame rate | ≥ 95% | < 90% |
| End-to-end update rate | ≥ 15 Hz | < 10 Hz |
| Position jitter while fixating | ≤ 0.35° RMS | > 0.75° RMS |

Do not declare “commercial-grade” from the nine calibration points' training error. The primary endpoint is the held-out, translated-head angular error. Also report the worst participant and failure rate; average-only results hide exactly the robustness problem this experiment targets.

## Interpretation order

1. Reject any method that improves mean error by clipping predictions but worsens P95 or edge targets.
2. If iris depth is noisy, compare its frame-to-frame depth standard deviation with the inter-eye proxy. A hybrid median/Kalman fusion is the next experiment.
3. If `metric_raycast` is biased in one screen region but stable across head positions, camera-to-screen extrinsics—not gaze estimation—are the likely bottleneck.
4. If every method drifts together, inspect UniGaze gaze direction under head rotation and upgrade the gaze backbone before adding more compensation logic.

## Research basis

- [Google MediaPipe Iris](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/): single-RGB metric distance from iris diameter, reported mean relative error about 4.3% in the original evaluation.
- [WebEyeTrack (2025)](https://github.com/RedForestAI/WebEyeTrack): metric face reconstruction and gaze origins for commodity webcams.
- [UniGaze](https://github.com/ut-vision/UniGaze): normalized gaze prediction; its video example transforms the predicted vector back through the inverse normalization rotation.
- [OpenCV `solvePnP`](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html): full-frame fixed camera intrinsics are required for meaningful translation. A face-centred crop with focal length proportional to crop size suppresses the translation signal.
