
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset

class SignDataset(Dataset):
    def __init__(self, kp_dir, json_path, top_classes):
        self.X, self.y = [], []

        with open(json_path) as f:
            data = json.load(f)

        self.label_map = {g:i for i,g in enumerate(top_classes)}

        video_to_label = {}
        for entry in data:
            gloss = entry["gloss"]
            if gloss not in self.label_map:
                continue
            for inst in entry["instances"]:
                video_to_label[inst["video_id"]] = self.label_map[gloss]

        for file in os.listdir(kp_dir):
            if not file.endswith(".npy"):
                continue
            vid = file.replace(".npy", "")
            if vid in video_to_label:
                self.X.append(np.load(os.path.join(kp_dir, file)))
                self.y.append(video_to_label[vid])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.num_classes = len(top_classes)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
