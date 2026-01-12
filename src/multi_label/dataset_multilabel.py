import os
import torch
from torch.utils.data import Dataset
from PIL import Image

INSTRUMENTS = [
    "cel", "cla", "flu", "gac", "gel",
    "org", "pia", "sax", "tru", "vio"
]

class IRMASMultiLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for inst in INSTRUMENTS:
            inst_dir = os.path.join(root_dir, inst)
            for img in os.listdir(inst_dir):
                self.samples.append((os.path.join(inst_dir, img), inst))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, inst = self.samples[idx]

        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)

        label = torch.zeros(len(INSTRUMENTS))
        label[INSTRUMENTS.index(inst)] = 1.0  # simulated multi-label

        return image, label
