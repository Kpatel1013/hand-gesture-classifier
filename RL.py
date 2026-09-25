import argparse
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)


def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def make_landmarker() -> HandLandmarker:
    path = _project_dir() / "models" / "hand_landmarker.task"
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return HandLandmarker.create_from_options(options)


def detect_hands(landmarker: HandLandmarker, frame_bgr: np.ndarray, timestamp_ms: int):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def draw_hands_on_frame(frame_bgr: np.ndarray, hand_landmarks_list: list, point_color: tuple[int, int, int] = (255, 200, 0)):
    spec_pt = drawing_utils.DrawingSpec(color=point_color, thickness=2, circle_radius=2)
    spec_ln = drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
    for lm in hand_landmarks_list:
        drawing_utils.draw_landmarks(
            frame_bgr,
            lm,
            HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=spec_pt,
            connection_drawing_spec=spec_ln,
        )
