import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import DeepfakeDataset
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2️⃣ Transforms (same as train.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3️⃣ Load dataset and split again
dataset = DeepfakeDataset(
    real_dir="data/train/real",
    fake_dir="data/train/fake",
    real_json="data/real_cifake_preds.json",
    fake_json="data/fake_cifake_preds.json",
    transform=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_ds = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# 4️⃣ Load trained model
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)
model.load_state_dict(torch.load("output/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# 5️⃣ Collect predictions and labels
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = torch.sigmoid(model(imgs))
        preds = (outputs > 0.5).int().squeeze()
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

# 6️⃣ Compute metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("\n📊 Validation Metrics:")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1-score : {f1*100:.2f}%")