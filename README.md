🎵 Audio-Based Musical Instrument Recognition using CNNs

This project implements an end-to-end audio-based musical instrument recognition system using Mel spectrograms and Convolutional Neural Networks (CNNs).
Raw audio signals are transformed into time–frequency representations, enabling deep learning models to identify the musical instrument present in an audio clip.

The system includes a trained CNN backend and a Streamlit-based frontend for interactive inference.

🚀 Project Highlights

🎧 Audio classification using deep learning

📊 Mel spectrogram–based feature extraction

🧠 CNN model implemented in PyTorch

🖥️ Interactive Streamlit web interface

🥇 Top-1 and Top-3 predictions with confidence

📈 Model evaluation with standard metrics

📌 Project Overview

Input: Audio file (.wav / .mp3)

Preprocessing: Mel Spectrogram generation using Librosa

Model: Convolutional Neural Network (CNN)

Framework: PyTorch

Output:

Predicted instrument (Top-1)

Top-3 predictions with confidence scores

Frontend: Streamlit web app for real-time inference

🎼 Instruments Classified

The model is trained using the IRMAS dataset and supports classification of the following instruments:

Cel – Cello

Cla – Clarinet

Flu – Flute

GAc – Acoustic Guitar

GEl – Electric Guitar

Org – Organ

Pia – Piano

Sax – Saxophone

Tru – Trumpet

Vio – Violin

🗂 Project Structure
instrument_recognition/
│
├── data/                 # Raw audio dataset (IRMAS)
├── spectrograms/         # Generated Mel spectrograms
├── models/               # Trained CNN models (.pth)
│
├── src/
│   ├── preprocess.py     # Audio preprocessing & spectrogram generation
│   ├── model.py          # CNN architecture
│   ├── train.py          # Model training script
│   ├── inference.py      # Model inference logic
│   ├── confusion_matrix.py
│   ├── classification_reports.py
│
├── test_audio/            # Sample audio files for testing
├── app.py                 # Streamlit frontend
├── requirements.txt       # Python dependencies
└── README.md

⚙️ Technologies Used

Python 3.10+

PyTorch

Librosa

NumPy

Pandas

Matplotlib

Scikit-learn

Streamlit

📊 Model Evaluation

The trained model was evaluated using standard classification metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Per-class accuracy

Top-3 prediction analysis

🖥️ Frontend (Streamlit App)

The Streamlit interface allows users to:

Upload an audio file

Listen to the uploaded audio

Run real-time instrument prediction

View:

Predicted instrument

Relative confidence score

Top-3 predictions table

Confidence scores are normalized across top predictions to improve interpretability for users.

🧠 Key Insight

This is not an image classification project.

Although CNNs are used, the input is audio, not images.
Spectrograms act as an intermediate time–frequency representation, allowing CNNs to learn discriminative audio features for instrument recognition.

📌 Limitations

Model confidence may be lower for:

Noisy audio

Short clips

Multiple instruments playing simultaneously

No real-time audio capture (file-based inference only)

🚧 Future Improvements

Improve accuracy using deeper CNN architectures

Apply data augmentation (time stretching, pitch shifting)

Use transfer learning on audio-specific models

Add real-time microphone input

Deploy on cloud platforms (Hugging Face Spaces / Render)

👤 Author

Ansh Goyal
B.Tech Electronics & Communication Engineering
AI / Machine Learning Enthusiast