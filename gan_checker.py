import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Dummy CNN model (replace with real pretrained model later)
class FakeImageDetector(nn.Module):
    def __init__(self):
        super(FakeImageDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 2)  # 2 classes: real or fake
        )

    def forward(self, x):
        return self.net(x)

model = FakeImageDetector()
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def check_if_image_fake(image: Image.Image) -> str:
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        return "Fake" if predicted.item() == 1 else "Real"