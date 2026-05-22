import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# === SegNet Model ===
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
        return self.decoder(self.encoder(x))

# === Color map ===
COLORS = {
    0: (0, 0, 0),        # Background
    1: (0, 0, 255),      # Sidewalk
    2: (255, 0, 0),      # Road
    3: (0, 255, 0)       # Crosswalk
}

# === Load model ===
model = SegNet(num_classes=4)
model.load_state_dict(torch.load("segnet_epoch_0086.pth", map_location='cpu'))
model.eval()

# === Dataset path and transform ===
root_dir = os.path.join(os.environ["USERPROFILE"], "Desktop", "signet_dataset")
val_img_dir = os.path.join(root_dir, "test", "images")
filenames = sorted(os.listdir(val_img_dir))
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# === Globals for viewer ===
current_index = 0
img_pil = None
probs_np = None
preds = None
current_filename = None

# === Setup plot ===
fig, ax = plt.subplots(figsize=(7, 6))
plt.subplots_adjust(bottom=0.25)
img_display = ax.imshow(np.zeros((1024, 1024, 3), dtype=np.uint8))
ax.set_xticks([])  # hide ticks but allow zoom
ax.set_yticks([])
title = ax.set_title("Loading...")

# === Slider ===
slider_ax = plt.axes([0.25, 0.1, 0.5, 0.03])
conf_slider = Slider(slider_ax, 'Confidence', 0.0, 1.0, valinit=0.2, valstep=0.01)

# === Buttons ===
back_ax = plt.axes([0.1, 0.02, 0.15, 0.06])
next_ax = plt.axes([0.75, 0.02, 0.15, 0.06])
back_button = Button(back_ax, '⬅ Back')
next_button = Button(next_ax, 'Next ➡')

# === Load a single image and run prediction ===
def load_image(index):
    global img_pil, probs_np, preds, current_filename

    filename = filenames[index]
    img_path = os.path.join(val_img_dir, filename)
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((1024, 1024))
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output.squeeze(0), dim=0)
        preds = torch.argmax(probs, dim=0).numpy()

    img_pil = img_resized
    probs_np = probs.numpy()
    current_filename = filename

# === Update display from confidence slider ===
def render():
    threshold = conf_slider.val
    mask = np.zeros((1024, 1024, 3), dtype=np.uint8)

    for y in range(1024):
        for x in range(1024):
            class_id = preds[y, x]
            confidence = probs_np[class_id, y, x]
            if confidence > threshold:
                mask[y, x] = COLORS[class_id]

    mask_img = Image.fromarray(mask)
    blended = Image.blend(img_pil, mask_img, alpha=0.5)
    img_display.set_data(blended)
    title.set_text(f"{current_filename} | Confidence > {threshold:.2f}")
    fig.canvas.draw_idle()

# === Button & slider callbacks ===
def on_slider_change(val):
    render()

def on_next(event):
    global current_index
    if current_index < len(filenames) - 1:
        current_index += 1
        load_image(current_index)
        render()

def on_back(event):
    global current_index
    if current_index > 0:
        current_index -= 1
        load_image(current_index)
        render()

conf_slider.on_changed(on_slider_change)
next_button.on_clicked(on_next)
back_button.on_clicked(on_back)

# === Init ===
load_image(current_index)
render()
plt.show()
