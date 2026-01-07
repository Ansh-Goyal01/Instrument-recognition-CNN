import os
import sys
import torch
import librosa
import numpy as np

# ---------------- FIX IMPORT PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(BASE_DIR)

from model import InstrumentCNN

# ---------------- CONFIG ----------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "instrument_cnn_best.pth")
AUDIO_PATH = os.path.join(PROJECT_ROOT, "test_audio", "[cla][cla]0150__2.wav")

CLASSES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio']
SR = 22050
N_MELS = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------
model = InstrumentCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- AUDIO → MEL ----------------
y, sr = librosa.load(AUDIO_PATH, sr=SR, mono=True)

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=N_MELS
)

mel_db = librosa.power_to_db(mel, ref=np.max)

# Normalize (same as training)
mel_db = (mel_db - mel_db.mean()) / mel_db.std()

# Shape: (1, 1, n_mels, time)
tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

# ---------------- INFERENCE ----------------
with torch.no_grad():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top3 = torch.topk(probs, 3)

print("\n🎵 Top-3 Predictions:")
for i in range(3):
    idx = top3.indices[i].item()
    print(f"{i+1}. {CLASSES[idx]} ({top3.values[i].item():.3f})")
