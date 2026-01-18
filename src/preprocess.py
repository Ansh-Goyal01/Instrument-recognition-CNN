import os
import librosa
import numpy as np
import torch

print("🔥 preprocess.py STARTED")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "spectrograms")

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# TRAINING PREPROCESSING
# =========================
def create_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    np.save(save_path, mel_db)


# =========================
# INFERENCE PREPROCESSING
# =========================
def preprocess_audio(audio_path):
    """
    Used during inference
    Returns tensor of shape (1, 1, H, W)
    """
    y, sr = librosa.load(audio_path, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    mel_db = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0)

    return mel_db.float()


# =========================
# DATASET GENERATION
# =========================
if __name__ == "__main__":
    instruments = os.listdir(DATA_DIR)

    for instrument in instruments:
        inst_path = os.path.join(DATA_DIR, instrument)
        save_inst_path = os.path.join(SAVE_DIR, instrument)

        if not os.path.isdir(inst_path):
            continue

        os.makedirs(save_inst_path, exist_ok=True)

        for file in os.listdir(inst_path):
            if file.endswith(".wav"):
                audio_file = os.path.join(inst_path, file)
                spec_file = os.path.join(
                    save_inst_path,
                    file.replace(".wav", ".npy")
                )
                create_spectrogram(audio_file, spec_file)

    print("✅ ALL DONE")
