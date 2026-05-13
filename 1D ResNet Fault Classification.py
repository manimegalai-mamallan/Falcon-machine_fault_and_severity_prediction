#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import json
import random
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ================= CONFIG =================
class CFG:
    data_dir = "."
    output_dir = "resnet1d_fault_outputs"

    epochs = 10
    batch_size = 32
    lr = 1e-3

    target_len = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"


Path(CFG.output_dir).mkdir(exist_ok=True)


# ================= SEED =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ================= SIGNAL PROCESSING =================
def resample(signal, target_len):
    signal = np.asarray(signal, dtype=np.float32)
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def preprocess(signal):
    signal = np.array(signal, dtype=np.float32)

    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    # make shape (channels, length)
    if signal.ndim == 2 and signal.shape[0] > signal.shape[1]:
        signal = signal.T

    processed = []
    for ch in signal:
        processed.append(resample(ch, CFG.target_len))
    signal = np.stack(processed, axis=0)

    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-8
    signal = (signal - mean) / std

    return signal.astype(np.float32)


def extract_signal_from_record(record):
    signal = record.get("signal", None)
    if signal is None:
        signal = record.get("data", None)
    if signal is None:
        signal = record.get("values", None)

    if signal is None:
        raise ValueError("No signal/data/values key found in JSONL record")

    if isinstance(signal, dict):
        # try X, Y, Z
        keys = [k for k in ["X", "Y", "Z", "x", "y", "z"] if k in signal]
        if keys:
            signal = [signal[k] for k in keys]
        else:
            signal = list(signal.values())

    return signal


# ================= LOAD DATA =================
def load_data():
    X, y = [], []

    files = glob(os.path.join(CFG.data_dir, "*.jsonl"))
    print("Files found:", files)

    if len(files) == 0:
        raise FileNotFoundError("No .jsonl files found in current folder")

    label_map = {
        "Bearing": "Bearing Fault",
        "Mechanical": "Mechanical Looseness",
        "Unbalanced": "Unbalance",
        "Normal": "Normal",
        "Misalignment": "Misalignment"
    }

    for file in files:
        raw_label = os.path.basename(file).split("_")[0]
        label = label_map.get(raw_label, raw_label)

        count = 0
        with open(file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    signal = extract_signal_from_record(record)
                    signal = preprocess(signal)

                    # force 3 channels for Streamlit compatibility
                    if signal.shape[0] == 1:
                        signal = np.vstack([signal, np.zeros_like(signal), np.zeros_like(signal)])
                    elif signal.shape[0] == 2:
                        signal = np.vstack([signal, np.zeros((1, signal.shape[1]), dtype=np.float32)])
                    elif signal.shape[0] > 3:
                        signal = signal[:3]

                    X.append(signal)
                    y.append(label)
                    count += 1

                except Exception as e:
                    print(f"Skipping line {line_num} in {os.path.basename(file)}: {e}")

        print(f"Loaded {count} samples from {os.path.basename(file)} as '{label}'")

    return np.array(X, dtype=np.float32), np.array(y)


# ================= DATASET =================
class FaultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ================= MODEL =================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.l1 = ResBlock(64, 64, stride=1)
        self.l2 = ResBlock(64, 128, stride=2)
        self.l3 = ResBlock(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


# ================= LOAD + SPLIT =================
print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir("."))

X, y_raw = load_data()

print("X shape:", X.shape)
print("y shape:", y_raw.shape)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
class_names = label_encoder.classes_.tolist()

print("Class names:", class_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_loader = DataLoader(FaultDataset(X_train, y_train), batch_size=CFG.batch_size, shuffle=True)
test_loader = DataLoader(FaultDataset(X_test, y_test), batch_size=CFG.batch_size, shuffle=False)


# ================= TRAIN =================
model = ResNet1D(num_classes=len(class_names)).to(CFG.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

print("Training started on:", CFG.device)

for epoch in range(CFG.epochs):
    model.train()
    total_loss = 0.0

    for Xb, yb in train_loader:
        Xb = Xb.to(CFG.device)
        yb = yb.to(CFG.device)

        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(train_loader), 1)
    print(f"Epoch {epoch + 1}/{CFG.epochs} - Loss: {avg_loss:.4f}")


# ================= SAVE MODEL =================
model_path = os.path.join(CFG.output_dir, "best_resnet1d_fault_classifier.pt")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "label_encoder_classes": class_names,
        "config": {
            "model_channels": [64, 128, 256],
            "dropout": 0.3
        }
    },
    model_path
)

print("Model saved at:", model_path)


# ================= EVALUATION =================
model.eval()
preds, trues = [], []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(CFG.device)
        out = model(Xb)
        p = torch.argmax(out, dim=1).cpu().numpy()

        preds.extend(p)
        trues.extend(yb.numpy())

acc = accuracy_score(trues, preds)
f1 = f1_score(trues, preds, average="macro")

print("\nAccuracy:", acc)
print("Macro F1:", f1)

print("\nClassification Report:\n")
print(classification_report(trues, preds, target_names=class_names, zero_division=0))

cm = confusion_matrix(trues, preds)
print("\nConfusion Matrix:\n", cm)

# confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest")
fig.colorbar(im, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)

threshold = cm.max() / 2.0 if cm.size > 0 else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.tight_layout()
plt.savefig(os.path.join(CFG.output_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.show()


# In[ ]:




