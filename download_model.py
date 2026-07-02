from torchvision import models
import torch
import torch.nn as nn

print("Downloading ResNet50...")
model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 38)
torch.save(model.state_dict(), 'models/plant_disease_model.pth')
print("Model saved!")