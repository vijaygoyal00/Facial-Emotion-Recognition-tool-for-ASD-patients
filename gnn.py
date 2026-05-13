!pip install torch_geometric

from google.colab import drive
drive.mount('/content/drive')

import os

def load_dataset(folder_path):
    data = []
    labels = []
    label_map = {}

    label_id = 0

    for emotion in os.listdir(folder_path):
        emotion_path = os.path.join(folder_path, emotion)

        if not os.path.isdir(emotion_path):
            continue

        label_map[label_id] = emotion

        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            data.append(img_path)
            labels.append(label_id)

        label_id += 1

    return data, labels, label_map

base_path = "/content/drive/MyDrive/Facial Emotion Recognition Dataset for Children wi/Facial Emotion Recognition Dataset for Children wi/Autism emotion recogition dataset/Autism emotion recogition dataset"

train_path = base_path + "/train"
test_path = base_path + "/test"

import os

train_path = None

for root, dirs, files in os.walk("/content/drive/MyDrive"):
    if "train" in dirs:
        train_path = os.path.join(root, "train")
        print("Found train path:", train_path)
        break

print(train_path)

train_data, train_labels, label_map = load_dataset(train_path)

print("Classes:", label_map)
print("Total images:", len(train_data))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import cv2
import numpy as np

def extract_landmarks(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    landmarks = []

    selected_indices = [

    # Left eye
    33, 133, 160, 159, 158, 144, 145, 153,

    # Right eye
    362, 263, 387, 386, 385, 373, 374, 380,

    # Left eyebrow
    70, 63, 105, 66, 107,

    # Right eyebrow
    300, 293, 334, 296, 336,

    # Mouth outer
    61, 146, 91, 181, 84,
    17, 314, 405, 321, 375, 291,

    # Mouth inner
    78, 95, 88, 178,
    87, 14, 317, 402, 318, 324, 308,

    # Nose
    1, 2, 98, 327, 168
]

    for idx in selected_indices:

        lm = face_landmarks.landmark[idx]

        landmarks.append([lm.x, lm.y, lm.z])

    landmarks = np.array(landmarks)

    # Normalize X
    landmarks[:,0] = (
        landmarks[:,0] - landmarks[:,0].min()
    ) / (
        landmarks[:,0].max() - landmarks[:,0].min()
    )

    # Normalize Y
    landmarks[:,1] = (
        landmarks[:,1] - landmarks[:,1].min()
    ) / (
        landmarks[:,1].max() - landmarks[:,1].min()
    )

    return landmarks

from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data

def create_graph(landmarks, label, k=4):

    # Node features
    x = torch.tensor(
        landmarks,
        dtype=torch.float
    )

    # KNN graph creation
    nbrs = NearestNeighbors(
        n_neighbors=k,
        algorithm='ball_tree'
    ).fit(landmarks)

    distances, indices = nbrs.kneighbors(landmarks)

    edge_index = []

    for i in range(len(indices)):

        for j in indices[i]:

            if i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    y = torch.tensor([label], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

!pip install protobuf==5.28.3
!pip install tensorflow==2.18.0
!pip install mediapipe==0.10.14

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import mediapipe as mp

print(mp.__version__)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True
)

print("MediaPipe working")

import mediapipe as mp

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True
)

graph_dataset = []

for img_path, label in zip(train_data, train_labels):

    landmarks = extract_landmarks(img_path)

    if landmarks is None:
        continue

    graph = create_graph(
        landmarks,
        label,
        k=4
    )

    graph_dataset.append(graph)

print("Graphs created:", len(graph_dataset))

torch.save(graph_dataset, "graph_dataset.pt")

torch.save(
    graph_dataset,
    "/content/drive/MyDrive/graph_dataset.pt"
)

from sklearn.model_selection import train_test_split

train_graphs, test_graphs = train_test_split(
    graph_dataset,
    test_size=0.2,
    random_state=42
)

print("Train graphs:", len(train_graphs))
print("Test graphs:", len(test_graphs))

from torch_geometric.loader import DataLoader

train_loader = DataLoader(
    train_graphs,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_graphs,
    batch_size=16
)

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):

    def __init__(self, hidden_channels, num_classes):

        super(GCN, self).__init__()

        self.conv1 = GCNConv(3, hidden_channels)

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.lin = Linear(
            hidden_channels,
            num_classes
        )

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.3,
            training=self.training
        )

        x = self.conv2(x, edge_index)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.3,
            training=self.training
        )

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(
    hidden_channels=32,
    num_classes=len(label_map)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(device)

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)

train_losses = []

for epoch in range(50):

    model.train()
    total_loss = 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = criterion(out, batch.y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

import matplotlib.pyplot as plt

plt.plot(train_losses, marker='o')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.grid(True)

plt.show()

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        pred = out.argmax(dim=1)

        correct += int((pred == batch.y).sum())

        total += batch.y.size(0)

accuracy = correct / total

print(f"Test Accuracy: {accuracy:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        preds = out.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

print(classification_report(
    all_labels,
    all_preds,
    target_names=list(label_map.values())
))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=list(label_map.values()),
    yticklabels=list(label_map.values())
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

import torch

graph_dataset = torch.load(
    "/content/drive/MyDrive/graph_dataset.pt",
    weights_only=False
)

from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data

def create_graph(landmarks, label, k=4):

    # Node features
    x = torch.tensor(
        landmarks,
        dtype=torch.float
    )

    # KNN graph creation
    nbrs = NearestNeighbors(
        n_neighbors=k,
        algorithm='ball_tree'
    ).fit(landmarks)

    distances, indices = nbrs.kneighbors(landmarks)

    edge_index = []

    for i in range(len(indices)):

        for j in indices[i]:

            if i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    y = torch.tensor([label], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

import torch
from torch.nn import Linear

class GAT(torch.nn.Module):

    def __init__(self, hidden_channels, num_classes):

        super(GAT, self).__init__()

        self.gat1 = GATConv(
            3,
            hidden_channels,
            heads=4
        )

        self.gat2 = GATConv(
            hidden_channels * 4,
            hidden_channels,
            heads=4
        )

        self.lin = Linear(
            hidden_channels * 4,
            num_classes
        )

    def forward(self, x, edge_index, batch):

        x = self.gat1(x, edge_index)
        x = F.relu(x)

        x = self.gat2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x

device2 = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
model = GAT(
    hidden_channels=32,
    num_classes=len(label_map)
).to(device2)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = torch.nn.CrossEntropyLoss()

train_losses2 = []

for epoch in range(50):

    model.train()
    total_loss = 0

    for batch in train_loader:

        batch = batch.to(device2)

        optimizer.zero_grad()

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = criterion(out, batch.y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    train_losses2.append(avg_loss)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device2)

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        pred = out.argmax(dim=1)

        correct += int((pred == batch.y).sum())

        total += batch.y.size(0)

accuracy = correct / total

print(f"Test Accuracy: {accuracy:.4f}")

import matplotlib.pyplot as plt

plt.plot(train_losses2, marker='o')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAT Training Loss")

plt.grid(True)

plt.show()

from sklearn.metrics import classification_report

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        preds = out.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

print(classification_report(
    all_labels,
    all_preds,
    target_names=list(label_map.values())
))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=list(label_map.values()),
    yticklabels=list(label_map.values())
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("GAT Confusion Matrix")

plt.show()
