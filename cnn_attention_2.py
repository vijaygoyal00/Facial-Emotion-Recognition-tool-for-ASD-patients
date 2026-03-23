# cnn_se_face_emotion_folders.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# ----------------------------
# Squeeze-and-Excitation Block
# ----------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ----------------------------
# Emotion Model (CNN + SE Attention)
# ----------------------------
class EmotionModelSE(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.base = models.resnet18(weights="IMAGENET1K_V1")
        self.base.fc = nn.Identity()
        self.se = SEBlock(channel=512, reduction=16)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        features = self.base(x)
        features_reshaped = features.unsqueeze(-1).unsqueeze(-1)
        features_se = self.se(features_reshaped).squeeze(-1).squeeze(-1)
        output = self.classifier(features_se)
        return output
    
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=224, margin=10, device=device)

# class FaceCrop:
#     def __call__(self, img):
#         face = mtcnn(img)
#         if face is None:
#             return transforms.ToTensor()(img)  # fallback
#         return face

input_root = "FER_autism/dataset/train"
output_root = "FER_autism_faces/train"

for class_name in os.listdir(input_root):
    os.makedirs(os.path.join(output_root, class_name), exist_ok=True)

    for img_name in os.listdir(os.path.join(input_root, class_name)):
        img_path = os.path.join(input_root, class_name, img_name)
        img = Image.open(img_path).convert("RGB")

        face = mtcnn(img)

        if face is not None:
            face_img = transforms.ToPILImage()(face)
            face_img.save(os.path.join(output_root, class_name, img_name))

input_root_test = "FER_autism/dataset/test"    # path to test folder
output_root_test = "FER_autism_faces/test"

for class_name in os.listdir(input_root_test):
    os.makedirs(os.path.join(output_root_test, class_name), exist_ok=True)

    for img_name in os.listdir(os.path.join(input_root_test, class_name)):
        img_path = os.path.join(input_root_test, class_name, img_name)
        img = Image.open(img_path).convert("RGB")

        face = mtcnn(img)

        if face is not None:
            face_img = transforms.ToPILImage()(face)
            face_img.save(os.path.join(output_root_test, class_name, img_name))

lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
# ----------------------------
# Transforms
# ----------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# Datasets & DataLoaders
# ----------------------------
train_dataset = datasets.ImageFolder(output_root, transform=transform_train)
test_dataset = datasets.ImageFolder(output_root_test, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# Model, Loss, Optimizer
# ----------------------------
num_classes = len(train_dataset.classes)
model = EmotionModelSE(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

# ----------------------------
# Training Loop
# ----------------------------
def train_epoch():
    model.train()
    correct, total = 0, 0
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(train_loader), correct / total



def train_final():
    print("Phase 1: Training classifier...")
    for param in model.base.parameters():
        param.requires_grad = False

    for epoch in range(10):
        loss, acc = train_epoch()
        val_acc = eval()
        scheduler.step()

        print(f"[Phase1][Epoch {epoch+1}] Loss: {loss:.4f}, Train Acc: {acc:.4f}, Val Acc: {val_acc:.4f}")

    # Phase 2: Fine-tune entire model
    print("\nPhase 2: Fine-tuning full model...")
    for param in model.base.parameters():
        param.requires_grad = True

    for epoch in range(25):
        loss, acc = train_epoch()
        val_acc = eval()
        scheduler.step()

        print(f"[Phase2][Epoch {epoch+1}] Loss: {loss:.4f}, Train Acc: {acc:.4f}, Val Acc: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "emotion_model_mtcnn_resnet18_se25.pth")
    print("\nModel saved!")


def eval_infer():
    model = EmotionModelSE(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("cnnNoPretrain_attention_SE_haar_40.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0

    print("Running inference on test dataset...\n")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
        
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")

def eval():
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

train_final()



