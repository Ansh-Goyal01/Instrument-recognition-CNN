import os
import matplotlib.pyplot as plt

# ---------------- PATHS ----------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------- LOAD METRICS ----------------
# ⚠️ Import directly from train.py saved variables
from train import train_accs, val_accs, train_losses, val_losses, EPOCHS

epochs = range(1, EPOCHS + 1)

# ---------------- ACCURACY ----------------
plt.figure()
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.grid()
plt.savefig(os.path.join(PLOT_DIR, "accuracy_curve.png"))
plt.close()

# ---------------- LOSS ----------------
plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid()
plt.savefig(os.path.join(PLOT_DIR, "loss_curve.png"))
plt.close()

print("✅ Accuracy & Loss curves saved in /plots")
