import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import json

# Settings
DATA_DIR = "dataset/plantvillage dataset/color"
EPOCH1_MODEL = "models/plant_disease_epoch2.pth"
MODEL_PATH = "models/plant_disease_model.pth"
BATCH_SIZE = 32
EPOCHS = 3  # 4 more epochs on top of epoch 1
IMG_SIZE = 224
DEVICE = torch.device('cpu')

print("Setting up...")

# Stronger augmentation for real-world images
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                          saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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
with open('models/class_names.json', 'w') as f:
    json.dump(full_dataset.classes, f)

# Split 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=0)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")

# Load ResNet50 FROM epoch 1 checkpoint
print("Loading epoch 1 model...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
model.load_state_dict(torch.load(EPOCH1_MODEL, map_location='cpu'))
model = model.to(DEVICE)

# Unfreeze last 2 layers for better learning
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 and fc
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

print("Unfroze layer4 + fc for deeper learning!")

# Optimizer with lower learning rate
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001}
])
criterion = nn.CrossEntropyLoss()

# Training loop
print("\nStarting training from epoch 3...")
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
            print(f"Epoch {epoch+3}/5 | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.1f}%")

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

    val_acc = 100.*val_correct/val_total
    print(f"\n✅ Epoch {epoch+3}/5 Complete | Val Accuracy: {val_acc:.1f}%\n")

    # Save after every epoch
    epoch_path = f'models/plant_disease_epoch{epoch+3}.pth'
    torch.save(model.state_dict(), epoch_path)
    print(f"💾 Saved: {epoch_path}")

# Save final model
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Final model saved!")