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

WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP = 3, 6, 10, 14, 18

GESTURES = ["OPEN", "FIST", "POINT", "PEACE", "THUMBS_UP", "NONE"]
G_TO_A = {g: i for i, g in enumerate(GESTURES)}
N_ACTIONS = len(GESTURES)


def _dist(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def finger_flags(lm, margin=0.02):
    """lm: indexable sequence of 21 landmarks with .x, .y."""
    wrist = (lm[WRIST].x, lm[WRIST].y)
    pairs = [
        (THUMB_TIP, THUMB_IP),
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    ]
    out = []
    for tip_idx, pip_idx in pairs:
        tip = (lm[tip_idx].x, lm[tip_idx].y)
        pip = (lm[pip_idx].x, lm[pip_idx].y)
        dt = _dist(tip, wrist)
        dp = _dist(pip, wrist)
        out.append(1 if dt > dp + margin else 0)
    return tuple(out)


def reference_gesture_from_flags(flags):
    t, i, m, r, p = flags
    if t and not i and not m and not r and not p:
        return G_TO_A["THUMBS_UP"]
    if not t and i and not m and not r and not p:
        return G_TO_A["POINT"]
    if not t and i and m and not r and not p:
        return G_TO_A["PEACE"]
    if sum(flags) == 0:
        return G_TO_A["FIST"]
    if sum(flags) >= 4:
        return G_TO_A["OPEN"]
    return G_TO_A["NONE"]


def state_from_flags(flags):
    return int("".join(str(b) for b in flags), 2)


class QGestureAgent:
    def __init__(self, n_states=32, n_actions=N_ACTIONS, lr=0.35, gamma=0.92, epsilon=0.25):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)

    def pick_action(self, state, train=True):
        if train and np.random.random() < self.epsilon:
            return np.random.randint(0, self.q.shape[1])
        return int(np.argmax(self.q[state]))

    def step(self, s, a, r, s_next):
        q_sa = self.q[s, a]
        backup = r + self.gamma * float(np.max(self.q[s_next]))
        self.q[s, a] = q_sa + self.lr * (backup - q_sa)

    def policy_action(self, state):
        return int(np.argmax(self.q[state]))
