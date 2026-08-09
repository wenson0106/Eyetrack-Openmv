"""Download and verify the model files required by the eye-tracking server."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unigaze_personalization.model import load_unigaze_b16


def main() -> None:
    face_landmarker = PROJECT_ROOT / "models" / "face_landmarker.task"
    if not face_landmarker.exists():
        raise FileNotFoundError(
            "models/face_landmarker.task is missing. Clone the complete repository again."
        )

    print("[1/2] MediaPipe face model OK:", face_landmarker)
    print("[2/2] Downloading UniGaze-B16 weights (about 350 MB)...")
    load_unigaze_b16("cpu")
    print("Model preparation complete.")


if __name__ == "__main__":
    main()
