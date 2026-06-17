from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "models"
MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        out_path = OUT_DIR / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"Modelo ja existe: {out_path}")
            continue

        print(f"Baixando modelo {filename}...")
        urlretrieve(url, out_path)
        print(f"Modelo salvo em: {out_path}")


if __name__ == "__main__":
    main()
