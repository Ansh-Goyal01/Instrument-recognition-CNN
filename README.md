# 🎵 Audio-Based Musical Instrument Recognition using CNNs

This project implements an **audio-based musical instrument recognition system** using **Mel spectrograms** and **Convolutional Neural Networks (CNNs)**.  
Raw audio signals are converted into time–frequency representations, enabling deep learning models to identify musical instruments present in an audio clip.

---

## 📌 Project Overview

- **Input**: Audio file (.wav)
- **Processing**: Mel Spectrogram generation using Librosa
- **Model**: CNN implemented in PyTorch
- **Output**: Predicted instrument class (Top-1 and Top-3 predictions)
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## 🎼 Instruments Classified

The model is trained on the **IRMAS dataset** and supports the following instruments:

cel - Cello
cla - Clarinet
flu - Flute
gac - Acoustic Guitar
gel - Electric Guitar
org - Organ
pia - Piano
sax - Saxophone
tru - Trumpet
vio - Violin


---

## 🗂 Project Structure



instrument_recognition/
│
├── data/ # Raw audio dataset (IRMAS)
├── spectrograms/ # Generated Mel spectrogram images
├── models/ # Saved trained models (.pth)
│
├── src/
│ ├── preprocess.py # Audio → spectrogram conversion
│ ├── model.py # CNN architecture
│ ├── train.py # Model training script
│ ├── test.py # Inference on new audio
│ ├── confusion_matrix.py # Evaluation & metrics
│
├── test_audio/ # Custom audio files for testing
├── venv/ # Python virtual environment
└── README.md


---

## ⚙️ Technologies Used

- Python 3.10+
- PyTorch
- Librosa
- NumPy
- Matplotlib
- Scikit-learn

📊 Model Evaluation

Confusion Matrix

Accuracy

Precision, Recall, F1-score

Top-3 Predictions with confidence scores

🧠 Key Insight

This is not an image classification project.
It is an audio classification system that leverages spectrograms as an intermediate representation to enable CNN-based feature learning.

📌 Future Improvements

Improve accuracy using deeper CNNs

Data augmentation (time stretching, pitch shifting)

Transfer learning

Real-time instrument recognition

👤 Author

Ansh Goyal
B.Tech ECE | AI/ML enthusiast
