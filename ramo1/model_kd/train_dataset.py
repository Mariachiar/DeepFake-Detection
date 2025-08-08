import torch
from torch.utils.data import Dataset
import os

class FaceForensicsClipDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.class_name = []

        # Cattura tutti i file .pt ricorsivamente
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".pt"):
                    path = os.path.join(root, file)

                    # Estrai la classe dalla path (es. nome sottocartella madre)
                    class_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
                    if class_name not in self.class_name:
                        self.class_name.append(class_name)

                    # Label: originale = 0, tutto il resto = 1
                    label = 0 if class_name == "original" else 1

                    self.samples.append((path, label))

        print(f"Trovati {len(self.samples)} file da {len(self.class_name)} classi: {self.class_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]
        clip = torch.load(clip_path)
        return clip, label
