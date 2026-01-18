import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader

from model import InstrumentCNN

# ---------------- PATHS ----------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

SPEC_DIR = os.path.join(BASE_DIR, "spectrograms")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "instrument_cnn_best.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- CLASSES ----------------
CLASSES = sorted([
    d for d in os.listdir(SPEC_DIR)
    if os.path.isdir(os.path.join(SPEC_DIR, d))
])

NUM_CLASSES = len(CLASSES)
print("Using classes:", CLASSES)

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

# ---------------- LOAD DATA ----------------
paths, labels = [], []

for idx, inst in enumerate(CLASSES):
    inst_dir = os.path.join(SPEC_DIR, inst)
    for f in os.listdir(inst_dir):
        if f.endswith(".npy"):
            paths.append(os.path.join(inst_dir, f))
            labels.append(idx)

print(f"Loaded {len(paths)} spectrograms")

dataset = SpectrogramDataset(paths, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ---------------- MODEL ----------------
model = InstrumentCNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        preds = model(x).argmax(1)
        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())

print(f"Predictions collected: {len(y_true)}")

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

plt.figure(figsize=(10, 8))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()

save_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
plt.savefig(save_path)
plt.close()

print(f"✅ Confusion matrix saved at: {save_path}")
