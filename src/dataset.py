import json
import os
from torch.utils.data import Dataset
from PIL import Image

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, real_json, fake_json, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform

        # Load label JSONs for real and fake images
        with open(real_json, 'r') as f:
            self.real_labels = json.load(f)
        with open(fake_json, 'r') as f:
            self.fake_labels = json.load(f)

        # Combine image paths and labels
        self.all_data = []
        for img_name in self.real_labels.keys():
            self.all_data.append((os.path.join(self.real_dir, img_name), 1))  # real = 1
        for img_name in self.fake_labels.keys():
            self.all_data.append((os.path.join(self.fake_dir, img_name), 0))  # fake = 0

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        img_path, label = self.all_data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label