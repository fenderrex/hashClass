#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# === SegNet model definition ===
class SegNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SegNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# === Custom Dataset ===
class SegNetDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.img_dir = os.path.join(root, split, 'images')
        self.lbl_dir = os.path.join(root, split, 'labels')
        self.filenames = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        lbl_path = os.path.join(self.lbl_dir, self.filenames[idx])

        image = Image.open(img_path).convert('RGB')
        label = Image.open(lbl_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        label_tensor = self.rgb_to_class(label)
        return image, label_tensor

    def rgb_to_class(self, label):
        r = label[0, :, :]
        g = label[1, :, :]
        b = label[2, :, :]

        class_map = torch.zeros_like(r, dtype=torch.long)
        class_map[(r < 0.1) & (g < 0.1) & (b < 0.1)] = 0  # black = background
        class_map[(r < 0.1) & (g < 0.1) & (b > 0.5)] = 1  # blue = sidewalk
        class_map[(r > 0.5) & (g < 0.1) & (b < 0.1)] = 2  # red = road
        class_map[(r < 0.1) & (g > 0.5) & (b < 0.1)] = 3  # green = crosswalk

        return class_map

# === Emoji progress bar function ===
def print_emoji_progress(epoch, batch_idx, total_batches, bar_length=30):
    progress = (batch_idx + 1) / total_batches
    filled_len = int(bar_length * progress)
    bar = '🟩' * filled_len + '⬜' * (bar_length - filled_len)
    print(f"\rEpoch {epoch+1}: {bar} {int(progress*100)}%", end='', flush=True)

# === Training Function ===
def train():
    root_dir = os.path.join(os.environ["USERPROFILE"], "Desktop", "signet_dataset")

    img_size = (1024, 1024)  # High resolution
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    train_set = SegNetDataset(root_dir, 'train', transform)
    val_set = SegNetDataset(root_dir, 'val', transform)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SegNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 300
    prev_loss = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print_emoji_progress(epoch, batch_idx, total_batches)

        avg_loss = total_loss / total_batches
        print()  # newline after progress bar

        if prev_loss is None or avg_loss <= prev_loss:
            print(f"✅ Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
        else:
            print(f"❌ Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f} (worse than previous)")

        prev_loss = avg_loss

        # Save every 10 epochs
        if (epoch + 1) % 2 == 0:
            save_path = f"segnet_epoch_{epoch+1:04d}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved checkpoint: {save_path}")

    print("🎯 Training complete.")
    torch.save(model.state_dict(), "segnet_final.pth")

if __name__ == "__main__":
    train()
