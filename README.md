# Hand Gesture Classifier

A webcam program that reads one hand with MediaPipe and trains a Q-learning agent to name the gesture. It tracks 21 hand landmarks, turns five fingers into a state, and learns which gesture that state is.

Gestures: open, fist, point, peace, and thumbs up. `NONE` is used when the pose does not match one of those.

## What it does

- Opens the webcam with OpenCV and runs the MediaPipe hand landmarker on each frame.
- Marks a finger as extended when its tip is farther from the wrist than its middle joint.
- Packs those five finger flags into one of 32 states.
- Trains a Q-table for about 2,500 steps. The reward is 1 when the chosen gesture matches the finger rule, and a small penalty otherwise.
- Saves the table to `q_gesture.npz` so the same policy can be loaded again without retraining.

The finger rules are the labels. The Q-learning loop is what the program practices during the live video.

## Run it

From this folder, with a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python RL.py
```

`models/hand_landmarker.task` has to sit next to `RL.py`. Training shows the camera window and quits after the step limit, or when you press `q`.

```bash
python RL.py --train-steps 2500 --epsilon 0.2 --save q_gesture.npz
```

## Stack

Python, OpenCV, MediaPipe, NumPy.
