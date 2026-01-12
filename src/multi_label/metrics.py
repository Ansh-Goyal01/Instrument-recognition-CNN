import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

def evaluate_multilabel(y_true, y_pred):
    print("Precision (macro):", precision_score(y_true, y_pred, average="macro"))
    print("Recall (macro):   ", recall_score(y_true, y_pred, average="macro"))
    print("F1-score (macro): ", f1_score(y_true, y_pred, average="macro"))
    print("Hamming Loss:     ", hamming_loss(y_true, y_pred))
