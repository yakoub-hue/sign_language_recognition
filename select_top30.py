
import os, json
from collections import Counter

KP_DIR = "data/keypoints"
JSON_PATH = "WLASL_v0.3.json"

with open(JSON_PATH) as f:
    data = json.load(f)

video_to_gloss = {}
for entry in data:
    gloss = entry["gloss"]
    for inst in entry["instances"]:
        video_to_gloss[inst["video_id"]] = gloss

counts = Counter()
for f in os.listdir(KP_DIR):
    if f.endswith(".npy"):
        vid = f.replace(".npy", "")
        if vid in video_to_gloss:
            counts[video_to_gloss[vid]] += 1

top30 = [g for g, _ in counts.most_common(30)]
print("Top 30 classes:", top30)
print("Counts:", {g: counts[g] for g in top30})
