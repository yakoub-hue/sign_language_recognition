import cv2
import json
import numpy as np
import mediapipe as mp
import os
from tqdm import tqdm

SEQ_LEN = 60
OUT_DIR = "data/keypoints"
VIDEO_DIR = "data/videos"

os.makedirs(OUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

with open("WLASL_v0.3.json", "r") as f:
    data = json.load(f)

saved = 0

for entry in tqdm(data):
    instances = entry["instances"]

    for inst in instances:
        if inst["split"] != "train":
            continue

        video_id = inst["video_id"]
        bbox = inst["bbox"]
        start = inst["frame_start"]
        end = inst["frame_end"]

        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx < start:
                continue
            if end != -1 and frame_idx > end:
                break

            x1, y1, x2, y2 = bbox
            h, w, _ = frame.shape

           
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = mp_hands.process(rgb)

            keypoints = np.zeros((126,), dtype=np.float32)

            if result.multi_hand_landmarks:
                for i, hand in enumerate(result.multi_hand_landmarks[:2]):
                    for j, lm in enumerate(hand.landmark):
                        keypoints[i*63 + j*3:(i*63 + j*3)+3] = [
                            lm.x, lm.y, lm.z
                        ]

            frames.append(keypoints)

        cap.release()

       
        frames = frames[:SEQ_LEN]
        while len(frames) < SEQ_LEN:
            frames.append(np.zeros((126,), dtype=np.float32))

        np.save(os.path.join(OUT_DIR, f"{video_id}.npy"), np.array(frames))
        saved += 1
