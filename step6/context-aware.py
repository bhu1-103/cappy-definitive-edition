import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Config ---
X_path = "step6/X_grouped.npy"
y_path = "step6/y_grouped.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4096
EPOCHS = 20
EMBED_DIM = 16
LR = 1e-3

# --- Load grouped data ---
X_grouped = np.load(X_path, allow_pickle=True)
y_grouped = np.load(y_path, allow_pickle=True)
n_scenarios = len(X_grouped)

# --- Flatten inputs and add scenario_id ---
X_all, y_all, s_id = [], [], []
for i, (x, y) in enumerate(zip(X_grouped, y_grouped)):
    if len(x) != len(y):
        continue
    X_all.append(x)
    y_all.append(y.reshape(-1, 1))
    s_id.append(np.full(len(x), i))

X_all = np.vstack(X_all).astype(np.float32)
y_all = np.vstack(y_all).astype(np.float32)
s_id = np.concatenate(s_id).astype(np.int64)

# --- Train/test split ---
X_train, X_test, y_train, y_test, sid_train, sid_test = train_test_split(
    X_all, y_all, s_id, test_size=0.2, random_state=42
)

# --- Dataset and Loader ---
class ScenarioDataset(Dataset):
    def __init__(self, X, y, s_id):
        self.X = X
        self.y = y
        self.s_id = s_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.s_id[idx], self.y[idx]

train_loader = DataLoader(ScenarioDataset(X_train, y_train, sid_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ScenarioDataset(X_test, y_test, sid_test), batch_size=BATCH_SIZE)

# --- Thick Context-Aware MLP ---
class ContextAwareMLP(nn.Module):
    def __init__(self, input_dim=5, scenario_count=n_scenarios, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(scenario_count, embed_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, sid):
        s_embed = self.embed(sid)
        x = torch.cat([x, s_embed], dim=1)
        return self.net(x)

# --- Init ---
model = ContextAwareMLP().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training ---
print(f"🧠 Training on {device} with {len(train_loader)} batches per epoch")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, sb, yb in train_loader:
        xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)

        pred = model(xb, sb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"📦 Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}")

# --- Evaluation ---
model.eval()
with torch.no_grad():
    total_test_loss = 0
    for xb, sb, yb in test_loader:
        xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
        pred = model(xb, sb)
        loss = loss_fn(pred, yb)
        total_test_loss += loss.item() * xb.size(0)

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    print(f"✅ Final Test MSE: {avg_test_loss:.4f}")

# --- Sample Prediction ---
print("\n🎯 Sample Predictions (Actual vs Predicted):")
model.eval()
with torch.no_grad():
    xb, sb, yb = next(iter(test_loader))
    xb, sb = xb.to(device), sb.to(device)
    pred = model(xb, sb).cpu().squeeze()
    yb = yb.squeeze()

    for i in range(20):
        gt = yb[i].item()
        pr = pred[i].item()
        print(f"GT: {gt:10.2f} | Pred: {pr:10.2f}")
torch.save(model.state_dict(), "step6/models/contextaware_thick_dropout.pt")
