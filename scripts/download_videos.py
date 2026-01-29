import json
import os
import requests
from tqdm import tqdm

JSON_PATH = "WLASL_v0.3.json"
OUT_DIR = "data/videos"

os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

downloaded = 0
failed = 0


for entry in tqdm(data):
    gloss = entry["gloss"]
    instances = entry["instances"]

    for inst in instances:
        if inst["split"] != "train":
            continue

        video_id = inst["video_id"]
        url = inst["url"]

        out_path = os.path.join(OUT_DIR, f"{video_id}.mp4")
        if os.path.exists(out_path):
            continue

        try:
            r = requests.get(url, timeout=15)
            with open(out_path, "wb") as f:
                f.write(r.content)
            downloaded += 1
        except Exception as e:
            print(f" Failed {video_id}: {e}")
            failed += 1

print(f" Downloaded: {downloaded}")
print(f" Failed: {failed}")