import torch
import librosa
import numpy as np
import os

from model import InstrumentCNN

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "instrument_cnn_best.pth")
AUDIO_PATH = os.path.join(BASE_DIR, "test_audio", "cla", "cla0150_2.wav")

SR = 22050
N_MELS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = sorted(os.listdir(os.path.join(BASE_DIR, "spectrograms")))

# ---------------- MODEL ----------------
model = InstrumentCNN(len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- PREPROCESS ----------------
y, sr = librosa.load(AUDIO_PATH, sr=SR)

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=N_MELS
)

mel_db = librosa.power_to_db(mel, ref=np.max)

# Normalize (same as training)
mel_db = (mel_db - mel_db.mean()) / mel_db.std()

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
