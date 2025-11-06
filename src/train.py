import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from dataset import DeepfakeDataset
from tqdm import tqdm
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Device setup (MPS for Mac)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomRotation(degrees=5),
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
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  # Binary classification
model = model.to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
best_val_acc = 0.0
patience = 2
no_improve_epochs = 0
epochs = 10
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
    val_acc = correct / total * 100
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}  Val Acc: {val_acc:.2f}%")

    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_epochs = 0
        torch.save(model.state_dict(), "output/best_model.pth")
        print("✅ Saved new best model!")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("🛑 Early stopping triggered!")
            break

# Save model
os.makedirs("output", exist_ok=True)
torch.save(model.state_dict(), "output/model.pth")
print("✅ Model saved to output/model.pth")