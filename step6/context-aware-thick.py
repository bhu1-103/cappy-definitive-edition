import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch.amp import autocast, GradScaler

# --- Config ---
X_path = "step6/X_grouped.npy"
y_path = "step6/y_grouped.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4096
EPOCHS = 20
EMBED_DIM = 8
LR = 1e-3
NUM_WORKERS = 4
MODEL_PATH = "step6/models/context_aware_thick.pt"
SCALER_PATH = "step6/models/scalers.npz"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# --- Load grouped data ---
X_grouped = np.load(X_path, allow_pickle=True)
y_grouped = np.load(y_path, allow_pickle=True)
n_scenarios = len(X_grouped)

# --- Flatten and add scenario IDs ---
X_all, y_all, s_id = [], [], []
for i, (x, y) in enumerate(zip(X_grouped, y_grouped)):
    if len(x) != len(y): continue
    X_all.append(x)
    y_all.append(y.reshape(-1, 1))
    s_id.append(np.full(len(x), i))

X_all = np.vstack(X_all).astype(np.float32)
y_all = np.vstack(y_all).astype(np.float32)
s_id = np.concatenate(s_id).astype(np.int64)

# --- Scaling ---
x_scaler = StandardScaler()
y_transform = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)

X_all = x_scaler.fit_transform(X_all)
y_all = y_transform.transform(y_all)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test, sid_train, sid_test = train_test_split(
    X_all, y_all, s_id, test_size=0.2, random_state=42
)

# --- Dataset ---
class ScenarioDataset(Dataset):
    def __init__(self, X, y, s_id):
        self.X = X
        self.y = y
        self.s_id = s_id

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.s_id[idx], self.y[idx]

# --- DataLoader ---
train_loader = DataLoader(
    ScenarioDataset(X_train, y_train, sid_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=2
)
test_loader = DataLoader(
    ScenarioDataset(X_test, y_test, sid_test),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=2
)

# --- Model ---
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class ContextAwareMLP(nn.Module):
    def __init__(self, input_dim=5, scenario_count=n_scenarios, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(scenario_count, embed_dim)
        self.input_proj = nn.Linear(input_dim + embed_dim, 128)
        self.resblocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.output = nn.Linear(128, 1)

    def forward(self, x, sid):
        s_embed = self.embed(sid)
        x = torch.cat([x, s_embed], dim=1)
        x = self.input_proj(x)
        x = self.resblocks(x)
        return self.output(x)

# --- Training ---
model = ContextAwareMLP().to(device)
loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler(device='cuda')

print(f"🧠 Training on {device} with {len(train_loader)} batches per epoch")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, sb, yb in train_loader:
        xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            pred = model(xb, sb)
            loss = loss_fn(pred, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"📦 Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}")

# --- Evaluation ---
model.eval()
total_test_loss = 0
all_preds = []
all_targets = []
sample_ids = []

with torch.no_grad():
    for xb, sb, yb in test_loader:
        xb, sb = xb.to(device), sb.to(device)
        with autocast(device_type='cuda'):
            pred = model(xb, sb).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(yb.numpy())
        sample_ids.append(sb.cpu().numpy())

preds = np.vstack(all_preds)
targets = np.vstack(all_targets)
scenarios = np.concatenate(sample_ids)

# --- Inverse scaling for final MSE ---
preds_inv = y_transform.inverse_transform(preds)
targets_inv = y_transform.inverse_transform(targets)
mse = ((preds_inv - targets_inv) ** 2).mean()

print(f"✅ Final Test MSE (scaled): {mse:.4f}")

# --- Save model and scalers ---
torch.save(model.state_dict(), MODEL_PATH)
np.savez("step6/models/scalers.npz",
         x_mean=x_scaler.mean_,
         x_std=x_scaler.scale_)
print(f"💾 Saved model + scalers")

# --- Sample predictions ---
print("\n🎯 Sample Predictions (Actual vs Predicted):")
for i in range(20):
    print(f"GT: {targets_inv[i][0]:6.2f} | Pred: {preds_inv[i][0]:6.2f} | Scenario: {scenarios[i]}")
