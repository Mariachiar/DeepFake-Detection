import torch
from torch.utils.data import Dataset
import os

class FaceForensicsClipDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        for subfolder in os.listdir(root_dir):
            full_subfolder_path = os.path.join(root_dir, subfolder)
            if not os.path.isdir(full_subfolder_path):
                continue
            
            # Label assignment
            label = 0 if subfolder == "original" else 1  # oppure {"original": 0, "Deepfakes": 1, ...}[subfolder]
            
            for fname in os.listdir(full_subfolder_path):
                if fname.endswith(".pt"):
                    path = os.path.join(full_subfolder_path, fname)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]
        clip = torch.load(clip_path)
        return clip, label
