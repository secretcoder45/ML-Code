import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dataset import DeepfakeDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

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

#Load trained model
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)
model.load_state_dict(torch.load("output/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# 5️⃣ Collect probabilities and labels
y_true, y_prob = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([probs])
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().tolist())

y_true = np.array(y_true)
y_pred = (np.array(y_prob) > 0.5).astype(int)

# 6️⃣ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])

plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Deepfake Detection")
plt.grid(False)
plt.show()
plt.savefig("output/confusion_matrix.png", dpi=300)

# 7️⃣ ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
plt.savefig("output/ROC_curve.png", dpi=300, bbox_inches="tight")

# 8️⃣ Precision–Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
f1 = f1_score(y_true, y_pred)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color="purple", lw=2, label=f"F1 = {f1:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend(loc="lower left")
plt.show()
plt.savefig("output/pr_curve.png", dpi=300)