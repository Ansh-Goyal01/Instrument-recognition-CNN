import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from model import InstrumentCNN

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)   # go to project root

DATA_DIR = os.path.join(BASE_DIR, "data")
SPEC_DIR = os.path.join(BASE_DIR, "spectrograms")  # ✅ THIS is where preprocess.py saved data

MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "instrument_cnn.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "instrument_cnn_best.pth")

CLASSES = sorted([
    d for d in os.listdir(SPEC_DIR)
    if os.path.isdir(os.path.join(SPEC_DIR, d))
])

NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 32
EPOCHS = 30
LR = 0.0005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)
print("Classes:", CLASSES)

# ---------------- DATASET ----------------
class SpectrogramDataset(Dataset):
    def __init__(self, paths, labels, train=False):
        self.paths = paths
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        spec = np.load(self.paths[idx])
        spec = torch.tensor(spec).unsqueeze(0).float()

        if self.train:
            # ---- SpecAugment (moderate) ----
            # Time masking
            t = np.random.randint(5, 20)
            t0 = np.random.randint(0, spec.shape[2] - t)
            spec[:, :, t0:t0+t] = 0

            # Frequency masking
            f = np.random.randint(3, 10)
            f0 = np.random.randint(0, spec.shape[1] - f)
            spec[:, f0:f0+f, :] = 0

        return spec, self.labels[idx]

# ---------------- LOAD DATA ----------------
paths, labels = [], []

for idx, inst in enumerate(CLASSES):
    inst_dir = os.path.join(SPEC_DIR, inst)
    for f in os.listdir(inst_dir):
        paths.append(os.path.join(inst_dir, f))
        labels.append(idx)

train_p, val_p, train_l, val_l = train_test_split(
    paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_ds = SpectrogramDataset(train_p, train_l, train=True)
val_ds = SpectrogramDataset(val_p, val_l, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ---------------- MODEL ----------------
model = InstrumentCNN(NUM_CLASSES).to(DEVICE)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_l),
    y=train_l
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5
)

# ---------------- METRICS ----------------
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_acc = 0.0

# ---------------- TRAIN LOOP ----------------
for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # ---- VALIDATION ----
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_loss = total_loss / len(val_loader)
    val_acc = correct / total

    scheduler.step(val_acc)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    # ---- SAVE BEST MODEL ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"💾 Best model saved! Val Acc = {best_val_acc:.4f}")

# ---------------- SAVE FINAL MODEL ----------------
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Final model saved")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")

# ---------------- PLOTS ----------------
epochs = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "accuracy_curve.png"))
plt.close()

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "loss_curve.png"))
plt.close()

print("📊 Accuracy & Loss plots saved in /plots")

import json

metrics = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_acc": train_accs,
    "val_acc": val_accs
}

with open(os.path.join(PLOT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f)

print("📁 Metrics saved to plots/metrics.json")
