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
# class EmotionModelSE(nn.Module):
#     def __init__(self, num_classes=6):
#         super().__init__()
#         self.base = models.resnet18(weights='ResNet18_Weights.DEFAULT')
#         self.base.fc = nn.Identity()
#         self.se = SEBlock(channel=512, reduction=8)
#         #self.classifier = nn.Linear(512, num_classes)
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.25),
#             nn.Linear(512, num_classes)
#         )           
    
#     def forward(self, x):
#         features = self.base(x)
#         features_reshaped = features.unsqueeze(-1).unsqueeze(-1)
#         features_se = self.se(features_reshaped).squeeze(-1).squeeze(-1)
#         output = self.classifier(features_se)
#         return output

class EmotionModelSE(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        base = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        # Feature extractor (remove avgpool + fc)
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,   # (B, 2048, 7, 7)
        )

        # ✅ Proper SE placement
        self.se = SEBlock(channel=2048, reduction=16)

        self.pool = nn.AdaptiveAvgPool2d(1)

        # ✅ Slight dropout (not too strong)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)              # (B, 2048, 7, 7)
        x = self.se(x)                   # SE applied correctly
        x = self.pool(x).view(x.size(0), -1)  # (B, 2048)
        x = self.classifier(x)
        return x
    
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=224, margin=10, device=device)

output_root = "FER_autism/dataset/train"
output_root_test = "FER_autism/dataset/test"


lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
# ----------------------------
# Transforms
# ----------------------------
# transform_train = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(20),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3),
#     transforms.ToTensor(),
# ])

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
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

num_classes = len(train_dataset.classes)
model = EmotionModelSE(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)


from collections import Counter

counts = Counter(train_dataset.targets)


# ----------------------------
# Training Loop
# ----------------------------


def train_final(best_params):

    #lr = best_params["lr"]
    #batch_size = best_params["batch_size"]
    #weight_decay = best_params["weight_decay"]
    #label_smooth = best_params["label_smooth"]
    #mixup_alpha = best_params["mixup_alpha"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = EmotionModelSE(num_classes=len(train_dataset.classes)).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30
    )
    class_weights = [1.0 / counts[i] for i in range(len(counts))]
    class_weights = torch.tensor(class_weights)
    class_weights = class_weights / class_weights.sum() * len(counts)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.01)

    # Phase 1
    print("Phase 1: Training classifier...")
    for param in model.features.parameters():
        param.requires_grad = False

    for epoch in range(5):
        epoch_loss = 0
        correct = 0
        total = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            images, y_a, y_b, lam = mixup(images, labels, 0.3)

            optimizer.zero_grad()
            outputs = model(images)

            loss = lam * criterion(outputs, y_a) + (1-lam) * criterion(outputs, y_b)
            #loss = criterion(outputs, labels)
            loss.backward()
            epoch_loss+= loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            optimizer.step()
        loss = epoch_loss/len(train_loader)
        acc = correct/total
        print(f"[Phase1][Epoch {epoch+1}] Loss: {loss:.4f}, Train Acc: {acc:.4f}")
        scheduler.step()

    # Phase 2
    print("\nPhase 2: Fine-tuning full model...")
    for param in model.features.parameters():
        param.requires_grad = True

    for epoch in range(40):
        epoch_loss = 0
        correct = 0
        total = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            images, y_a, y_b, lam = mixup(images, labels, 0.3)

            optimizer.zero_grad()
            outputs = model(images)

            loss = lam * criterion(outputs, y_a) + (1-lam) * criterion(outputs, y_b)
            #loss = criterion(outputs, labels)
            loss.backward()
            epoch_loss+= loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            optimizer.step()
        loss = epoch_loss/len(train_loader)
        acc = correct/total
        print(f"[Phase1][Epoch {epoch+1}] Loss: {loss:.4f}, Train Acc: {acc:.4f}")
        scheduler.step()

    torch.save(model.state_dict(), "best_model_ResNet50_0tri_40epo_no_mtcnn.pth")
    print("🔥 Final model trained and saved!")




def eval_infer():
    model = EmotionModelSE(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_model_ResNet50_0tri_40epo_no_mtcnn.pth", map_location=device))
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

import optuna

# ----------------------------
# Mixup
# ----------------------------
def mixup(x, y, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1-lam) * x[index], y, y[index], lam

# ----------------------------
# Objective Function
# ----------------------------
def objective(trial):

    # 🔍 Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    label_smooth = trial.suggest_float("label_smooth", 0.05, 0.2)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.2, 0.5)

    # DataLoaders (IMPORTANT: recreate with batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = EmotionModelSE(num_classes=len(train_dataset.classes)).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # ----------------------------
    # Phase 1: Freeze backbone
    # ----------------------------
    for param in model.base.parameters():
        param.requires_grad = False

    for epoch in range(5):  # short for tuning
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Mixup
            #images, y_a, y_b, lam = mixup(images, labels, mixup_alpha)

            optimizer.zero_grad()
            outputs = model(images)

            #loss = lam * criterion(outputs, y_a) + (1-lam) * criterion(outputs, y_b)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

    # ----------------------------
    # Phase 2: Fine-tune
    # ----------------------------
    for param in model.base.parameters():
        param.requires_grad = True

    best_acc = 0

    for epoch in range(8):  # short tuning phase
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            #images, y_a, y_b, lam = mixup(images, labels, mixup_alpha)

            optimizer.zero_grad()
            outputs = model(images)

            #loss = lam * criterion(outputs, y_a) + (1-lam) * criterion(outputs, y_b)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        best_acc = max(best_acc, acc)

        # Early stopping for Optuna
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_acc

# ----------------------------
# Run Optuna
# ----------------------------
def tuning():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\n🔥 Best Hyperparameters:")
    print(study.best_params)
    return study.best_params

# hyper_params = {'lr': 0.00025893572722989426, 
#                 'batch_size': 16, 
#                 'weight_decay': 2.9016434084193717e-06, 
#                 'label_smooth': 0.05611713082800912, 
#                 'mixup_alpha': 0.2507001760847839}
hyper_params = {}
#hyper_params = tuning()
#train_final(hyper_params)

eval_infer()