from __future__ import annotations

import json
import math
import sys
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

try:
    MP_BASE_OPTIONS = mp.tasks.BaseOptions
    MP_FACE_LANDMARKER = mp.tasks.vision.FaceLandmarker
    MP_FACE_OPTIONS = mp.tasks.vision.FaceLandmarkerOptions
    MP_HAND_LANDMARKER = mp.tasks.vision.HandLandmarker
    MP_HAND_OPTIONS = mp.tasks.vision.HandLandmarkerOptions
    MP_RUNNING_MODE = mp.tasks.vision.RunningMode
except AttributeError:
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_tasks_vision

    MP_BASE_OPTIONS = mp_tasks_python.BaseOptions
    MP_FACE_LANDMARKER = mp_tasks_vision.FaceLandmarker
    MP_FACE_OPTIONS = mp_tasks_vision.FaceLandmarkerOptions
    MP_HAND_LANDMARKER = mp_tasks_vision.HandLandmarker
    MP_HAND_OPTIONS = mp_tasks_vision.HandLandmarkerOptions
    MP_RUNNING_MODE = mp_tasks_vision.RunningMode


ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "meme_profiles.json"
FACE_MODEL_PATH = ROOT / "assets" / "models" / "face_landmarker.task"
HAND_MODEL_PATH = ROOT / "assets" / "models" / "hand_landmarker.task"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FACE_HOLD_FRAMES = 10
TRAIT_KEYS = (
    "smile",
    "tilt",
    "center",
    "brightness",
    "wide_face",
    "hand",
    "shaka",
    "thumbs_up",
    "point",
    "open_hand",
    "mouth_open",
    "brow_up",
)
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
LEFT_EYE = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
RIGHT_EYE = (362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382)
LEFT_BROW = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
RIGHT_BROW = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
HAND_TRAIT_KEYS = ("hand", "shaka", "thumbs_up", "point", "open_hand")


@dataclass
class MemeProfile:
    name: str
    image_path: Path
    description: str
    traits: dict[str, float]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def load_profiles() -> list[MemeProfile]:
    with PROFILE_PATH.open("r", encoding="utf-8") as file:
        raw_profiles: list[dict[str, Any]] = json.load(file)

    profiles = []
    for item in raw_profiles:
        profiles.append(
            MemeProfile(
                name=item["name"],
                image_path=ROOT / item["image"],
                description=item["description"],
                traits={key: float(item["traits"].get(key, 0.0)) for key in TRAIT_KEYS},
            )
        )
    return profiles


def image_to_tk(image: np.ndarray | Image.Image, size: tuple[int, int]) -> ImageTk.PhotoImage:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#101115")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return ImageTk.PhotoImage(canvas)


def distance(a: dict[str, float], b: dict[str, float]) -> float:
    weights = {
        "smile": 1.45,
        "tilt": 1.9,
        "center": 0.8,
        "brightness": 0.35,
        "wide_face": 0.7,
        "hand": 0.75,
        "shaka": 2.4,
        "thumbs_up": 2.0,
        "point": 1.6,
        "open_hand": 1.0,
        "mouth_open": 1.35,
        "brow_up": 1.1,
    }
    total = 0.0
    for key in TRAIT_KEYS:
        total += ((a[key] - b[key]) ** 2) * weights[key]
    return math.sqrt(total)


class MemeFaceApp:
    def __init__(self, root: tk.Tk, camera_index: int = 0) -> None:
        self.root = root
        self.root.title("Meme Face Detector")
        self.root.geometry("1180x720")
        self.root.minsize(980, 620)
        self.root.configure(bg="#111217")

        self.profiles = load_profiles()
        if not FACE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo de rosto nao encontrado: {FACE_MODEL_PATH}\n"
                "Rode: python scripts/download_hand_model.py"
            )
        face_options = MP_FACE_OPTIONS(
            base_options=MP_BASE_OPTIONS(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=MP_RUNNING_MODE.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_face_blendshapes=True,
        )
        self.face_landmarker = MP_FACE_LANDMARKER.create_from_options(face_options)
        self.face_cascades = [
            cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")),
            cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml")),
            cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml")),
        ]
        self.face_cascades = [cascade for cascade in self.face_cascades if not cascade.empty()]
        self.smile_cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_smile.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))
        if not HAND_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo de maos nao encontrado: {HAND_MODEL_PATH}\n"
                "Rode: python scripts/download_hand_model.py"
            )
        hand_options = MP_HAND_OPTIONS(
            base_options=MP_BASE_OPTIONS(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=MP_RUNNING_MODE.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands_detector = MP_HAND_LANDMARKER.create_from_options(hand_options)
        self.frame_timestamp_ms = 0
        self.camera_index = camera_index
        self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self.last_match: MemeProfile | None = None
        self.last_score = 0.0
        self.last_features = {key: 0.0 for key in TRAIT_KEYS}
        self.last_update = 0.0
        self.last_face: tuple[int, int, int, int] | None = None
        self.last_face_features: dict[str, float] | None = None
        self.face_miss_count = 0
        self.last_hand_features = {key: 0.0 for key in HAND_TRAIT_KEYS}
        self.camera_photo: ImageTk.PhotoImage | None = None
        self.meme_photo: ImageTk.PhotoImage | None = None

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    def build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#111217")
        style.configure("Title.TLabel", background="#111217", foreground="#f4f5f7", font=("Segoe UI", 22, "bold"))
        style.configure("Text.TLabel", background="#111217", foreground="#c9ced8", font=("Segoe UI", 11))
        style.configure("Metric.TLabel", background="#181a22", foreground="#f4f5f7", font=("Segoe UI", 10, "bold"))

        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 16))
        ttk.Label(header, text="Meme Face Detector", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text=f"camera {self.camera_index} -> sinais do rosto -> meme mais parecido",
            style="Text.TLabel",
        ).pack(side="right")

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0)
        body.rowconfigure(0, weight=1)

        self.camera_label = tk.Label(body, bg="#0b0c10", bd=0)
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        self.meme_label = tk.Label(body, bg="#0b0c10", bd=0)
        self.meme_label.grid(row=0, column=1, sticky="nsew", padx=(0, 12))

        side = ttk.Frame(body, padding=14)
        side.grid(row=0, column=2, sticky="ns")

        self.match_title = ttk.Label(side, text="Procurando rosto...", style="Title.TLabel")
        self.match_title.pack(anchor="w", pady=(0, 8))
        self.match_desc = ttk.Label(side, text="Fique de frente para a camera.", style="Text.TLabel", wraplength=260)
        self.match_desc.pack(anchor="w", pady=(0, 18))

        self.score_label = ttk.Label(side, text="Match: 0%", style="Metric.TLabel", padding=10)
        self.score_label.pack(fill="x", pady=(0, 12))

        self.feature_labels: dict[str, ttk.Label] = {}
        labels = {
            "smile": "Sorriso",
            "tilt": "Cabeca inclinada",
            "center": "Centralizado",
            "brightness": "Iluminacao",
            "wide_face": "Proporcao do rosto",
            "hand": "Dedos/pose",
            "shaka": "Hang loose",
            "thumbs_up": "Joinha",
            "point": "Apontando",
            "open_hand": "Mao aberta",
            "mouth_open": "Boca aberta",
            "brow_up": "Sobrancelha",
        }
        for key, label in labels.items():
            item = ttk.Label(side, text=f"{label}: 0.00", style="Metric.TLabel", padding=10)
            item.pack(fill="x", pady=4)
            self.feature_labels[key] = item

        hint = ttk.Label(
            side,
            text="Dica: sorria, incline a cabeca ou faca uma pose neutra para ver o meme trocar.",
            style="Text.TLabel",
            wraplength=260,
        )
        hint.pack(anchor="w", pady=(18, 0))

    def analyze(
        self,
        frame: np.ndarray,
        hand_features: dict[str, float],
        timestamp_ms: int,
    ) -> tuple[np.ndarray, dict[str, float] | None]:
        face_features = self.analyze_face_with_mediapipe(frame, timestamp_ms)
        if face_features is None:
            face_features = self.analyze_face_with_haar(frame)

        if face_features is None and self.last_face_features and self.face_miss_count < FACE_HOLD_FRAMES:
            self.face_miss_count += 1
            features = self.last_face_features.copy()
            features.update(hand_features)
            if self.last_face:
                x, y, w, h = self.last_face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 190, 255), 2)
                cv2.putText(
                    frame,
                    "rosto previsto",
                    (x, max(28, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (120, 190, 255),
                    2,
                )
            return frame, features

        if face_features is None:
            self.last_face = None
            self.last_face_features = None
            self.face_miss_count = 0
            return frame, None

        self.face_miss_count = 0
        features = face_features.copy()
        features.update(hand_features)
        self.last_face_features = features.copy()
        return frame, features

    def analyze_face_with_mediapipe(self, frame: np.ndarray, timestamp_ms: int) -> dict[str, float] | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        blendshapes = self.blendshape_scores(result)
        bbox = self.face_bbox(landmarks, frame.shape[1], frame.shape[0])
        self.last_face = bbox
        self.draw_face_landmarks(frame, landmarks, frame.shape[1], frame.shape[0])

        x, y, w, h = bbox
        roi = cv2.cvtColor(frame[max(0, y) : y + h, max(0, x) : x + w], cv2.COLOR_BGR2GRAY)
        brightness = clamp(float(np.mean(roi)) / 255) if roi.size else 0.5

        left_eye = landmarks[33]
        right_eye = landmarks[263]
        eye_dx = max(abs(right_eye.x - left_eye.x), 1e-6)
        eye_dy = right_eye.y - left_eye.y
        tilt = clamp(abs(math.degrees(math.atan2(eye_dy, eye_dx))) / 35)

        face_center_x = x + w / 2
        center_offset = abs(face_center_x - frame.shape[1] / 2) / (frame.shape[1] / 2)
        center = clamp(1 - center_offset)
        wide_face = clamp((w / max(h, 1) - 0.58) / 0.5)

        mouth_gap = self.landmark_distance(landmarks[13], landmarks[14])
        mouth_width = self.landmark_distance(landmarks[61], landmarks[291])
        mouth_ratio = mouth_gap / max(mouth_width, 1e-6)

        smile = clamp(
            max(
                (blendshapes.get("mouthSmileLeft", 0.0) + blendshapes.get("mouthSmileRight", 0.0)) * 1.15,
                (mouth_width / max(self.landmark_distance(landmarks[10], landmarks[152]), 1e-6) - 0.32) * 2.2,
            )
        )
        mouth_open = clamp(max(blendshapes.get("jawOpen", 0.0) * 1.6, (mouth_ratio - 0.04) * 5.5))
        brow_up = clamp(
            max(
                blendshapes.get("browInnerUp", 0.0),
                blendshapes.get("browOuterUpLeft", 0.0),
                blendshapes.get("browOuterUpRight", 0.0),
            )
            * 1.8
        )

        cv2.rectangle(frame, (x, y), (x + w, y + h), (63, 220, 120), 2)
        cv2.putText(frame, "face mesh", (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (63, 220, 120), 2)

        return {
            "smile": smile,
            "tilt": tilt,
            "center": center,
            "brightness": brightness,
            "wide_face": wide_face,
            "mouth_open": mouth_open,
            "brow_up": brow_up,
        }

    def analyze_face_with_haar(self, frame: np.ndarray) -> dict[str, float] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = self.find_face(gray)
        if face is None:
            return None

        self.last_face = face
        x, y, w, h = face
        roi_gray = gray[y : y + h, x : x + w]
        frame_h, frame_w = frame.shape[:2]

        mouth_gray = roi_gray[int(h * 0.45) :, :]
        smiles = self.smile_cascade.detectMultiScale(
            mouth_gray,
            scaleFactor=1.8,
            minNeighbors=24,
            minSize=(35, 20),
        )
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=8, minSize=(18, 18))

        smile_area = max((sw * sh for _, _, sw, sh in smiles), default=0)
        smile = clamp(smile_area / max(w * h * 0.09, 1))
        center_offset = abs((x + w / 2) - (frame_w / 2)) / (frame_w / 2)
        center = clamp(1 - center_offset)
        brightness = clamp(float(np.mean(roi_gray)) / 255)
        wide_face = clamp((w / max(h, 1) - 0.65) / 0.45)
        tilt = self.estimate_tilt(eyes, w, h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 80), 2)
        cv2.putText(frame, "rosto opencv", (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 80), 2)

        return {
            "smile": smile,
            "tilt": tilt,
            "center": center,
            "brightness": brightness,
            "wide_face": wide_face,
            "mouth_open": 0.0,
            "brow_up": 0.0,
        }

    @staticmethod
    def blendshape_scores(result: Any) -> dict[str, float]:
        if not getattr(result, "face_blendshapes", None):
            return {}
        return {
            category.category_name: float(category.score)
            for category in result.face_blendshapes[0]
        }

    @staticmethod
    def landmark_distance(a: Any, b: Any) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    @staticmethod
    def face_bbox(landmarks: Any, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        x1 = int(max(0, min(xs) * frame_w))
        y1 = int(max(0, min(ys) * frame_h))
        x2 = int(min(frame_w - 1, max(xs) * frame_w))
        y2 = int(min(frame_h - 1, max(ys) * frame_h))
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def draw_face_landmarks(frame: np.ndarray, landmarks: Any, frame_w: int, frame_h: int) -> None:
        def point(index: int) -> tuple[int, int]:
            item = landmarks[index]
            return int(item.x * frame_w), int(item.y * frame_h)

        def draw_path(indices: tuple[int, ...], color: tuple[int, int, int], close: bool = False) -> None:
            points = [point(index) for index in indices]
            for start, end in zip(points, points[1:]):
                cv2.line(frame, start, end, color, 1, cv2.LINE_AA)
            if close and len(points) > 2:
                cv2.line(frame, points[-1], points[0], color, 1, cv2.LINE_AA)

        draw_path(FACE_OVAL, (90, 190, 120), close=True)
        draw_path(LEFT_EYE, (120, 200, 180), close=True)
        draw_path(RIGHT_EYE, (120, 200, 180), close=True)
        draw_path(LEFT_BROW, (90, 230, 90))
        draw_path(RIGHT_BROW, (90, 230, 90))
        draw_path(LIPS, (80, 180, 255), close=True)

    def find_face(self, gray: np.ndarray) -> tuple[int, int, int, int] | None:
        candidates: list[tuple[int, int, int, int]] = []
        for cascade in self.face_cascades:
            faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(72, 72))
            candidates.extend(tuple(int(value) for value in face) for face in faces)

        flipped = cv2.flip(gray, 1)
        frame_w = gray.shape[1]
        for cascade in self.face_cascades:
            faces = cascade.detectMultiScale(flipped, scaleFactor=1.12, minNeighbors=4, minSize=(72, 72))
            for fx, fy, fw, fh in faces:
                candidates.append((int(frame_w - fx - fw), int(fy), int(fw), int(fh)))

        if not candidates:
            return None
        return max(candidates, key=lambda face: face[2] * face[3])

    @staticmethod
    def estimate_tilt(eyes: np.ndarray, face_w: int, face_h: int) -> float:
        if len(eyes) < 2:
            return 0.0
        sorted_eyes = sorted(eyes, key=lambda eye: eye[2] * eye[3], reverse=True)[:2]
        centers = [(x + w / 2, y + h / 2) for x, y, w, h in sorted_eyes]
        centers.sort(key=lambda point: point[0])
        dy = abs(centers[1][1] - centers[0][1])
        return clamp((dy / max(face_h, 1)) * 8)

    def estimate_hand_presence(self, frame: np.ndarray, timestamp_ms: int) -> dict[str, float]:
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.hands_detector.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            self.last_hand_features = {
                key: (0.0 if self.last_hand_features[key] < 0.08 else self.last_hand_features[key] * 0.86)
                for key in HAND_TRAIT_KEYS
            }
            return self.last_hand_features.copy()

        raw = {key: 0.0 for key in HAND_TRAIT_KEYS}
        for index, landmarks in enumerate(result.hand_landmarks):
            is_left = False
            if result.handedness and index < len(result.handedness):
                is_left = result.handedness[index][0].category_name == "Left"

            fingers = self.fingers_state(landmarks, is_left=is_left)
            open_fingers = sum(fingers)
            wrist_y = landmarks[0].y
            tips_above_wrist = sum(1 for tip in (4, 8, 12, 16, 20) if landmarks[tip].y < wrist_y)
            score = clamp((open_fingers * 0.18) + (tips_above_wrist * 0.06) + 0.12)
            raw["hand"] = max(raw["hand"], score)
            raw["shaka"] = max(raw["shaka"], self.gesture_shaka(fingers))
            raw["thumbs_up"] = max(raw["thumbs_up"], self.gesture_thumbs_up(fingers, landmarks))
            raw["point"] = max(raw["point"], self.gesture_point(fingers))
            raw["open_hand"] = max(raw["open_hand"], 1.0 if open_fingers >= 4 else 0.0)

            self.draw_hand_landmarks(frame, landmarks, frame_w, frame_h, fingers)
            xs = [int(point.x * frame_w) for point in landmarks]
            ys = [int(point.y * frame_h) for point in landmarks]
            label_x = max(8, min(xs))
            label_y = max(24, min(ys) - 8)
            gesture = self.gesture_label(raw)
            cv2.putText(
                frame,
                f"{gesture} dedos: {open_fingers} {fingers}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (80, 220, 255),
                2,
                cv2.LINE_AA,
            )

        if len(result.hand_landmarks) >= 2:
            raw["hand"] = clamp(raw["hand"] + 0.22)

        self.last_hand_features = {
            key: max(raw[key], self.last_hand_features[key] * 0.55 + raw[key] * 0.45)
            for key in HAND_TRAIT_KEYS
        }
        return self.last_hand_features.copy()

    @staticmethod
    def draw_hand_landmarks(frame: np.ndarray, landmarks: Any, frame_w: int, frame_h: int, fingers: list[int]) -> None:
        points = [(int(point.x * frame_w), int(point.y * frame_h)) for point in landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (120, 200, 180), 1, cv2.LINE_AA)
        for index, point in enumerate(points):
            color = (80, 240, 80) if index in (4, 8, 12, 16, 20) and fingers[[4, 8, 12, 16, 20].index(index)] else (80, 170, 255)
            cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)

    @staticmethod
    def fingers_state(landmarks: Any, is_left: bool) -> list[int]:
        wrist = landmarks[0]
        thumb_open = (
            MemeFaceApp.landmark_distance(landmarks[4], landmarks[5])
            > MemeFaceApp.landmark_distance(landmarks[3], landmarks[5]) * 1.18
            and MemeFaceApp.landmark_distance(landmarks[4], wrist)
            > MemeFaceApp.landmark_distance(landmarks[2], wrist) * 1.08
        )
        fingers = [1 if thumb_open else 0]
        for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
            extended = (
                MemeFaceApp.landmark_distance(landmarks[tip], wrist)
                > MemeFaceApp.landmark_distance(landmarks[pip], wrist) * 1.08
                and MemeFaceApp.landmark_distance(landmarks[tip], landmarks[mcp])
                > MemeFaceApp.landmark_distance(landmarks[pip], landmarks[mcp]) * 1.05
            )
            fingers.append(1 if extended else 0)
        return fingers

    @staticmethod
    def gesture_shaka(fingers: list[int]) -> float:
        thumb, index, middle, ring, pinky = fingers
        if thumb and pinky and not index and not middle and not ring:
            return 1.0
        if thumb and pinky and sum(fingers) <= 3:
            return 0.7
        return 0.0

    @staticmethod
    def gesture_thumbs_up(fingers: list[int], landmarks: Any) -> float:
        thumb, index, middle, ring, pinky = fingers
        thumb_tip_above_wrist = landmarks[4].y < landmarks[0].y
        if thumb and not index and not middle and not ring and not pinky and thumb_tip_above_wrist:
            return 1.0
        if thumb and sum(fingers) <= 2:
            return 0.55
        return 0.0

    @staticmethod
    def gesture_point(fingers: list[int]) -> float:
        thumb, index, middle, ring, pinky = fingers
        if index and not middle and not ring and not pinky:
            return 1.0 if not thumb else 0.75
        return 0.0

    @staticmethod
    def gesture_label(features: dict[str, float]) -> str:
        labels = {
            "shaka": "hang loose",
            "thumbs_up": "joinha",
            "point": "apontando",
            "open_hand": "mao aberta",
        }
        key, value = max(((key, features[key]) for key in labels), key=lambda item: item[1])
        return labels[key] if value >= 0.55 else "mao"

    def choose_meme(self, features: dict[str, float]) -> tuple[MemeProfile, float]:
        ranked = sorted(
            ((profile, distance(features, profile.traits)) for profile in self.profiles),
            key=lambda item: item[1],
        )
        best, dist = ranked[0]
        score = clamp(1 - dist / 1.6)
        return best, score

    def update_match(self, features: dict[str, float] | None) -> None:
        if features is None:
            self.match_title.configure(text="Sem rosto")
            self.match_desc.configure(text="Aproxime o rosto da camera para comecar a comparacao.")
            self.score_label.configure(text="Match: 0%")
            return

        now = time.time()
        if self.last_match is None or now - self.last_update > 0.35:
            self.last_match, self.last_score = self.choose_meme(features)
            self.last_features = features
            self.last_update = now

        self.match_title.configure(text=self.last_match.name)
        self.match_desc.configure(text=self.last_match.description)
        self.score_label.configure(text=f"Match: {self.last_score * 100:.0f}%")
        for key, label in self.feature_labels.items():
            title = label.cget("text").split(":")[0]
            label.configure(text=f"{title}: {self.last_features[key]:.2f}")

        meme_image = Image.open(self.last_match.image_path).convert("RGB")
        self.meme_photo = image_to_tk(meme_image, (430, 520))
        self.meme_label.configure(image=self.meme_photo)

    def tick(self) -> None:
        ok, frame = self.camera.read()
        if not ok:
            placeholder = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 16, dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Camera nao encontrada",
                (85, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (230, 230, 230),
                2,
            )
            frame = placeholder
            features = None
        else:
            frame = cv2.flip(frame, 1)
            timestamp_ms = self.next_timestamp_ms()
            hand_features = self.estimate_hand_presence(frame, timestamp_ms)
            frame, features = self.analyze(frame, hand_features, timestamp_ms)

        self.camera_photo = image_to_tk(frame, (430, 520))
        self.camera_label.configure(image=self.camera_photo)
        self.update_match(features)
        self.root.after(30, self.tick)

    def next_timestamp_ms(self) -> int:
        now_ms = int(time.monotonic() * 1000)
        self.frame_timestamp_ms = max(self.frame_timestamp_ms + 1, now_ms)
        return self.frame_timestamp_ms

    def close(self) -> None:
        self.face_landmarker.close()
        self.hands_detector.close()
        self.camera.release()
        self.root.destroy()


def list_cameras(max_index: int = 5) -> None:
    print("Testando cameras disponiveis...")
    found = False
    for index in range(max_index + 1):
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        ok, frame = camera.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            print(f"camera {index}: OK ({width}x{height})")
            found = True
        else:
            print(f"camera {index}: nao abriu")
        camera.release()

    if not found:
        print("Nenhuma camera respondeu. Confira permissao da camera no Windows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector de rosto que compara expressoes com memes.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indice da camera. Normalmente 0 = notebook, 1 ou 2 = webcam externa.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Testa indices de camera de 0 a 5 e mostra quais abriram.",
    )
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    missing_assets = [profile.image_path for profile in load_profiles() if not profile.image_path.exists()]
    if missing_assets:
        print("Assets de memes nao encontrados. Rode: python scripts/generate_sample_memes.py")
        for path in missing_assets:
            print(f"- {path}")
        sys.exit(1)

    root = tk.Tk()
    MemeFaceApp(root, camera_index=args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()
