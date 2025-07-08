import torch
import torch.nn as nn
import numpy as np
import os

# --- Config ---
X_PATH = "step6/X_grouped_sta.npy"
Y_PATH = "step6/y_grouped_sta.npy"
STA_MODEL_PATH = "step6/models/contextaware_sta.pt"
EMBED_DIM = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Data ---
X_grouped = np.load(X_PATH, allow_pickle=True)
y_grouped = np.load(Y_PATH, allow_pickle=True)
n_scenarios = len(X_grouped)

# --- Model Definition ---
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

# --- Load Model ---
model = ContextAwareMLP().to(DEVICE)
model.load_state_dict(torch.load(STA_MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Predict AP throughput by summing STA predictions ---
print("🔁 Predicting AP throughput by summing STA predictions...\n")

for scenario_id in range(n_scenarios):
    X = X_grouped[scenario_id]
    y = y_grouped[scenario_id]

    node_types = X[:, 0].astype(int)
    sta_indices = np.where(node_types == 1)[0]  # Select STAs only

    if len(sta_indices) == 0:
        print(f"⚠️ Scenario {scenario_id} skipped - No STAs found.")
        continue

    sta_x = X[sta_indices]
    sta_sid = np.full((len(sta_x),), scenario_id, dtype=np.int64)

    sta_tensor = torch.tensor(sta_x, dtype=torch.float32).to(DEVICE)
    sid_tensor = torch.tensor(sta_sid, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        sta_preds = model(sta_tensor, sid_tensor).squeeze().cpu().numpy()

    predicted_ap_total = sta_preds.sum()
    actual_ap_total = y.sum()

    print(f"📡 Scenario {scenario_id}: Predicted = {predicted_ap_total:.2f} Mbps | Actual = {actual_ap_total:.2f} Mbps")
