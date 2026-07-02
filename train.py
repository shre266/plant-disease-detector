import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# Settings
DATA_DIR = "dataset/plantvillage dataset/color"
MODEL_PATH = "models/plant_disease_model.pth"
BATCH_SIZE = 32
EPOCHS = 5
IMG_SIZE = 224
DEVICE = torch.device('cpu')

print("Setting up...")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
print("Loading dataset...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
print(f"Total images: {len(full_dataset)}")
print(f"Classes: {len(full_dataset.classes)}")

# Save class names
import json
with open('models/class_names.json', 'w') as f:
    json.dump(full_dataset.classes, f)
print("Class names saved!")

# Split 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")

# Load pretrained ResNet50
print("Loading ResNet50...")
model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
model = model.to(DEVICE)

# Freeze all layers except final
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.1f}%")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    print(f"\n✅ Epoch {epoch+1} Complete | Val Accuracy: {100.*val_correct/val_total:.1f}%\n")

# Save after every epoch
    epoch_path = f'models/plant_disease_epoch{epoch+1}.pth'
    torch.save(model.state_dict(), epoch_path)
    print(f"💾 Model saved: {epoch_path}")

# Save final model
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Final model saved to {MODEL_PATH}")