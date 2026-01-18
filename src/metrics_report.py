import os
import torch
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

from model import InstrumentCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(BASE_DIR, "spectrograms")
MODEL_PATH = os.path.join(BASE_DIR, "models", "instrument_cnn_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = sorted(os.listdir(SPEC_DIR))

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
    inst_dir = os.path.join(SPEC_DIR, inst)
    for f in os.listdir(inst_dir):
        paths.append(os.path.join(inst_dir, f))
        labels.append(idx)

loader = DataLoader(SpectrogramDataset(paths, labels), batch_size=32)

model = InstrumentCNN(len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        preds = model(x).argmax(1)
        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASSES))
