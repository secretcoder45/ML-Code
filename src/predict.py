# src/predict.py
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# Dataset for test images 
class DeepfakeTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_files = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    # transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load test dataset only from data/test/
test_dataset = DeepfakeTestDataset("data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)
model.load_state_dict(torch.load("output/model.pth", map_location=device))
model = model.to(device)
model.eval()

# Predict
results = []

with torch.no_grad():
    for imgs, names in tqdm(test_loader, desc="Predicting"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        if probs.ndim == 0:
            probs = [float(probs)]

        for name, prob in zip(names, probs):
            index = int(os.path.splitext(name)[0])
            pred = "real" if prob > 0.5 else "fake"
            results.append({
                "index": index,
                "prediction": pred
            })

# Sort and save
results = sorted(results, key=lambda x: x["index"])
os.makedirs("output", exist_ok=True)
with open("output/submission.json", "w") as f:
    json.dump(results, f, indent=4)

print("Predictions saved to output/submission.json")