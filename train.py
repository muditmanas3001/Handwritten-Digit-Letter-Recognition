import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Transforms ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ── Dataset (same 30% subset logic as before) ────────────────────────────────
train_dataset = datasets.EMNIST(root='./data', split='letters', train=True,  download=True, transform=transform)
test_dataset  = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

full_dataset = ConcatDataset([train_dataset, test_dataset])
total_len    = len(full_dataset)

portion_size = int(0.9 * total_len)
unused_size  = total_len - portion_size

subset_30_percent, _ = random_split(
    full_dataset,
    [portion_size, unused_size],
    generator=torch.Generator().manual_seed(42)
)
print(f"Subset size (90% of total): {len(subset_30_percent)}")

# ── NEW: Split into train (80%) / val (10%) / test (10%) ─────────────────────
train_size = int(0.80 * len(subset_30_percent))
val_size   = int(0.10 * len(subset_30_percent))
test_size  = len(subset_30_percent) - train_size - val_size

train_subset, val_subset, test_subset = random_split(
    subset_30_percent,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train size : {len(train_subset)}")
print(f"Val size   : {len(val_subset)}")
print(f"Test size  : {len(test_subset)}")

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_subset,  batch_size=64, shuffle=False)

# ── Model (unchanged) ─────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2)
        self.fc1   = nn.Linear(16 * 7 * 7, 256)
        self.fc2   = nn.Linear(256, 64)
        self.fc3   = nn.Linear(64, 27)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleCNN()
print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ── Training loop with validation loss tracking ───────────────────────────────
num_epochs   = 10
train_losses = []
val_losses   = []

for epoch in range(num_epochs):
    # — Train —
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # — Validate —
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs   = model(images)
            loss      = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")

# ── Plot 1: Train vs Validation Loss Curve ────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss',      color='steelblue')
plt.plot(range(1, num_epochs + 1), val_losses,   marker='s', label='Validation Loss', color='tomato')
plt.title('Train vs Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('train_val_loss_curve.png', dpi=150)
plt.close()
print("Saved: train_val_loss_curve.png")

# ── Test Accuracy ─────────────────────────────────────────────────────────────
model.eval()
correct = 0
total   = 0
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs        = model(images)
        _, predicted   = torch.max(outputs, 1)
        total         += labels.size(0)
        correct       += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ── Plot 2: Confusion Matrix ──────────────────────────────────────────────────
LETTERS      = [chr(i + 64) for i in range(1, 27)]   # A–Z
cm           = confusion_matrix(all_labels, all_preds, labels=list(range(1, 27)))
cm_norm      = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # row-normalised

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=LETTERS,
    yticklabels=LETTERS,
    linewidths=0.4,
    ax=ax,
    vmin=0, vmax=1
)
ax.set_title('Confusion Matrix (Row-Normalised)', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label',      fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# ── Save model ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "model/pytorch_ocr.pth")
print("Model saved: pytorch_ocr.pth")