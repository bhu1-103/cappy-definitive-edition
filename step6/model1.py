import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Load grouped data
X_grouped = np.load("step6/X_grouped.npy", allow_pickle=True)
y_grouped = np.load("step6/y_grouped.npy", allow_pickle=True)

# Split scenarios
n_scenarios = len(X_grouped)
train_idx, test_idx = train_test_split(np.arange(n_scenarios), test_size=0.2, random_state=42)

# Flatten scenario-wise groups
X_train = np.concatenate([X_grouped[i] for i in train_idx])
y_train = np.concatenate([y_grouped[i] for i in train_idx])
X_test  = np.concatenate([X_grouped[i] for i in test_idx])
y_test  = np.concatenate([y_grouped[i] for i in test_idx])

# Torch datasets
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

# Define model
class Net(nn.Module):
    def __init__(self, in_features=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(in_features=5).to(device)

# Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
batch_size = 1024

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

print(f"🧠 Training on {device} with {len(train_dl)} batches per epoch")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    avg_loss = running_loss / len(train_ds)
    print(f"📦 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).cpu().squeeze()
    mse = ((preds - y_test.squeeze())**2).mean().item()
    print(f"✅ Final Test MSE: {mse:.4f}")
