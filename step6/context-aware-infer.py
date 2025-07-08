import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from torch.utils.data import TensorDataset, DataLoader

# --- Config ---
SCENARIO_ID = 27372  # CHANGE THIS to the scenario you want
MODEL_PATH = "step6/models/context_aware_thick.pt"
X_path = "step6/X_grouped.npy"
y_path = "step6/y_grouped.npy"
SCALERS_PATH = "step6/models/scalers.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load data ---
X_grouped = np.load(X_path, allow_pickle=True)
y_grouped = np.load(y_path, allow_pickle=True)
n_scenarios = len(X_grouped)
assert 0 <= SCENARIO_ID < n_scenarios, "Invalid scenario ID."

X = X_grouped[SCENARIO_ID].astype(np.float32)
y = y_grouped[SCENARIO_ID].reshape(-1, 1).astype(np.float32)
sid = np.full(len(X), SCENARIO_ID, dtype=np.int64)

# --- Load scalers ---
scalers = np.load(SCALERS_PATH)
x_scaler = StandardScaler()
x_scaler.mean_ = scalers["x_mean"]
x_scaler.scale_ = scalers["x_std"]
y_transform = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)

X_scaled = x_scaler.transform(X)
y_scaled = y_transform.transform(y)

# --- Dataset ---
X_tensor = torch.tensor(X_scaled).to(device)
sid_tensor = torch.tensor(sid).to(device)
y_tensor = torch.tensor(y_scaled).to(device)

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
    def __init__(self, input_dim=5, scenario_count=n_scenarios, embed_dim=8):
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

# --- Load model ---
model = ContextAwareMLP().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Predict ---
with torch.no_grad():
    preds_scaled = model(X_tensor, sid_tensor)
    preds = y_transform.inverse_transform(preds_scaled.cpu().numpy())
    actual = y

# --- Print results ---
print(f"\n🎯 Scenario {SCENARIO_ID}: {len(actual)} samples\n")
print(f"{'Idx':<5} {'GT':>8} {'Pred':>8} {'Δ Error':>10}")
print("-" * 32)
for i, (gt, pred) in enumerate(zip(actual, preds)):
    delta = pred[0] - gt[0]
    print(f"{i:<5} {gt[0]:8.2f} {pred[0]:8.2f} {delta:10.2f}")

import matplotlib.pyplot as plt
plt.plot(actual, label="Ground Truth")
plt.plot(preds, label="Prediction")
plt.title(f"Scenario {SCENARIO_ID}")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.tight_layout()
plt.show()
