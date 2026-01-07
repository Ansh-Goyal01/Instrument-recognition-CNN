import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset_multilabel import IRMASMultiLabelDataset
from model_multilabel import InstrumentCNNMultiLabel
from metrics import multilabel_metrics, multilabel_accuracy

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.0005
THRESHOLD = 0.5
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------------- DATASET SPLITS ----------------
dataset = IRMASMultiLabelDataset(
    root_dir="../spectrograms",
    transform=transform
)

test_size = int(len(dataset) * TEST_SPLIT)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - test_size - val_size

train_ds, val_ds, test_ds = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
model = InstrumentCNNMultiLabel(num_classes=10).to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAINING ----------------
for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    # ---------- TRAIN ----------
    model.train()
    train_loss = 0
    y_true_train, y_pred_train = [], []

    for images, labels in train_loader:
        images = images.unsqueeze(1).to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = (torch.sigmoid(logits) > THRESHOLD).int()

        y_true_train.append(labels.cpu().numpy())
        y_pred_train.append(preds.cpu().numpy())

    y_true_train = np.vstack(y_true_train)
    y_pred_train = np.vstack(y_pred_train)

    train_acc = multilabel_accuracy(y_true_train, y_pred_train)

    # ---------- VALIDATION ----------
    model.eval()
    y_true_val, y_pred_val = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.unsqueeze(1).to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            preds = (torch.sigmoid(logits) > THRESHOLD).int()

            y_true_val.append(labels.cpu().numpy())
            y_pred_val.append(preds.cpu().numpy())

    y_true_val = np.vstack(y_true_val)
    y_pred_val = np.vstack(y_pred_val)

    val_acc = multilabel_accuracy(y_true_val, y_pred_val)
    val_metrics = multilabel_metrics(y_true_val, y_pred_val)

    # ---------- ACCURACY GAP ----------
    acc_gap_train_val = train_acc - val_acc

    # ---------- PRINT ----------
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Accuracy Gap (Train − Val): {acc_gap_train_val:.4f}")

    print("\n-- Validation Metrics --")
    print(f"Precision (Macro):   {val_metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):      {val_metrics['recall_macro']:.4f}")
    print(f"F1-score (Macro):    {val_metrics['f1_macro']:.4f}")
    print(f"F1-score (Weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"Hamming Loss:        {val_metrics['hamming_loss']:.4f}")

# ---------------- TEST EVALUATION ----------------
print("\n===== FINAL TEST EVALUATION =====")
model.eval()

y_true_test, y_pred_test = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.unsqueeze(1).to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        preds = (torch.sigmoid(logits) > THRESHOLD).int()

        y_true_test.append(labels.cpu().numpy())
        y_pred_test.append(preds.cpu().numpy())

y_true_test = np.vstack(y_true_test)
y_pred_test = np.vstack(y_pred_test)

test_acc = multilabel_accuracy(y_true_test, y_pred_test)
test_metrics = multilabel_metrics(y_true_test, y_pred_test)

# ---------- GAP ANALYSIS ----------
train_test_gap = train_acc - test_acc
val_test_gap = val_acc - test_acc

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Accuracy Gap (Train − Test): {train_test_gap:.4f}")
print(f"Accuracy Gap (Val − Test):   {val_test_gap:.4f}")

print("\n-- Test Metrics --")
print(f"Precision (Macro):   {test_metrics['precision_macro']:.4f}")
print(f"Recall (Macro):      {test_metrics['recall_macro']:.4f}")
print(f"F1-score (Macro):    {test_metrics['f1_macro']:.4f}")
print(f"F1-score (Weighted): {test_metrics['f1_weighted']:.4f}")
print(f"Hamming Loss:        {test_metrics['hamming_loss']:.4f}")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "irmas_multilabel_model.pth")
print("\n✅ Multi-label model saved successfully")
