import torch
from src.model import InstrumentCNN
from src.preprocess import preprocess_audio

# 🔴 CHANGE THIS TO MATCH YOUR TRAINING DATA ORDER
INSTRUMENT_LABELS = [
    "Piano",
    "Guitar",
    "Violin",
    "Drums",
    "Flute",
    "Saxophone",
    "Trumpet",
    "Cello",
    "Clarinet",
    "Harmonium"
]

DEVICE = "cpu"

# Load model
model = InstrumentCNN(num_classes=len(INSTRUMENT_LABELS))
model.load_state_dict(
    torch.load("models/instrument_cnn_best.pth", map_location=DEVICE)
)
model.eval()


def predict_instrument(audio_path):
    with torch.no_grad():
        features = preprocess_audio(audio_path)
        outputs = model(features)

        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=3)

    results = []
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        results.append({
            "instrument": INSTRUMENT_LABELS[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return results

# =========================
# TEST RUN
# =========================
if __name__ == "__main__":
    test_audio_path = "test_audio/[cla][cla]0150__2.wav"  # change if needed
    result = predict_instrument(test_audio_path)
    print(result)
