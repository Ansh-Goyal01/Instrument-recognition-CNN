import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from model import InstrumentCNN

# ---------------- PATHS ----------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "instrument_cnn_best.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- CLASSES ----------------
CLASSES = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

NUM_CLASSES = len(CLASSES)

# ---------------- DATASET ----------------
class SpectrogramDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        spec = np.load(self.paths[idx])
        spec = torch.tensor(spec).unsqueeze(0).float()
        return spec, self.labels[idx]

paths, labels = [], []

for idx, inst in enumerate(CLASSES):
    inst_dir = os.path.join(DATA_DIR, inst)
    for f in os.listdir(inst_dir):
        if f.endswith((".npy", ".npz")):
            paths.append(os.path.join(inst_dir, f))
            labels.append(idx)

loader = DataLoader(SpectrogramDataset(paths, labels), batch_size=32)

# ---------------- MODEL ----------------
model = InstrumentCNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

correct = [0] * NUM_CLASSES
total = [0] * NUM_CLASSES

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        preds = model(x).argmax(1).cpu().numpy()

        for label, pred in zip(y.numpy(), preds):
            total[label] += 1
            if label == pred:
                correct[label] += 1

accuracy = [c / t if t > 0 else 0 for c, t in zip(correct, total)]

# ---------------- PLOT ----------------
plt.figure(figsize=(10, 5))
plt.bar(CLASSES, accuracy)
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "per_class_accuracy.png"))
plt.close()

print("✅ Per-class accuracy plot saved")
