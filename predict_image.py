import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 8 * 8, 2)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCNN().to(device)
model.load_state_dict(torch.load('cat_dog_simple_model.pth', map_location=device))
model.eval()

class_names = ['cat', 'dog']

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, pred = torch.max(output, 1)
        class_name = class_names[pred.item()]
        print(f"Predicted Class: {class_name}")

        plt.imshow(image)
        plt.title(f"Prediction: {class_name}")
        plt.axis('off')
        plt.show()

predict_image(r'D:\Faizi\DeepLearning\Datasets\cat_dog_sub\test\cat\356.jpg')