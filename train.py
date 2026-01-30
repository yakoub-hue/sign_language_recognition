
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SignDataset
from model import SignLSTM

TOP30 =  ['drink', 'before', 'computer', 'go', 'now', 'shirt', 'thanksgiving', 'white', 'who', 'can', 'dance', 'hat', 'hearing', 'mother', 'wrong', 'bed', 'book', 'candy', 'color', 'cook', 'dog', 'family', 'orange', 'paper', 'play', 'same', 'study', 'want', 'apple', 'bird']

dataset = SignDataset("data/keypoints", "WLASL_v0.3.json", TOP30)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SignLSTM(dataset.num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1:02d} | Loss {total_loss:.3f} | Acc {acc:.2%}")
