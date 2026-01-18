import torch
from torchvision import transforms
from PIL import Image
from model_multilabel import InstrumentCNNMultiLabel

INSTRUMENTS = [
    "cel", "cla", "flu", "gac", "gel",
    "org", "pia", "sax", "tru", "vio"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = InstrumentCNNMultiLabel(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load("irmas_multilabel_model.pth", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

img_path = "test.png"  # put any spectrogram image here
image = Image.open(img_path).convert("L")
image = transform(image).unsqueeze(0).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(image)
    probs = torch.sigmoid(logits)[0]

threshold = 0.5
print("Predicted instruments:")
for inst, prob in zip(INSTRUMENTS, probs):
    if prob > threshold:
        print(f"{inst}  ({prob:.2f})")
