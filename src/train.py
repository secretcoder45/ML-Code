import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from dataset import DeepfakeDataset
from tqdm import tqdm
import os

# Device setup (MPS for Mac or CPU/GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
dataset = DeepfakeDataset(
    real_dir="data/train/real",
    fake_dir="data/train/fake",
    real_json="data/real_cifake_preds.json",
    fake_json="data/fake_cifake_preds.json",
    transform=transform
)

# Split into train and validation (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Define model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  # Binary classification
model = model.to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.unsqueeze(1).float().to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = torch.sigmoid(model(imgs)) > 0.5
            correct += (preds.squeeze().int() == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100
    print(f"Validation Accuracy: {acc:.2f}%")