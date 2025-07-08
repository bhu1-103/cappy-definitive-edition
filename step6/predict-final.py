import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# --- Config ---
MODEL_PATH = "step6/models/contextaware_sta.pt"
X_PATH = "step6/X_grouped_sta.npy"
Y_PATH = "step6/y_grouped_sta.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
class ContextAwareMLP(nn.Module):
    def __init__(self, input_dim=5, scenario_count=10000, embed_dim=16):
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

# --- Load data ---
X_grouped = np.load(X_PATH, allow_pickle=True)
y_grouped = np.load(Y_PATH, allow_pickle=True)

scenario_id = 0
X_scenario = X_grouped[scenario_id].astype(np.float32)
y_scenario = y_grouped[scenario_id].astype(np.float32).flatten()
sid_tensor = torch.full((len(X_scenario),), scenario_id, dtype=torch.long)

# --- Load model ---
model = ContextAwareMLP(scenario_count=len(X_grouped)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Predict ---
with torch.no_grad():
    x_tensor = torch.tensor(X_scenario, dtype=torch.float32).to(device)
    sid_tensor = sid_tensor.to(device)
    preds = model(x_tensor, sid_tensor).cpu().squeeze().numpy()

# --- Combine into dataframe ---
df = pd.DataFrame(X_scenario, columns=["node_type", "wlan_code", "x", "y", "z"])
df["predicted_throughput"] = preds
df["actual_throughput"] = y_scenario

# --- Group by wlan_code (AP) ---
grouped = df.groupby("wlan_code").agg(
    predicted_ap_throughput=("predicted_throughput", "sum"),
    actual_ap_throughput=("actual_throughput", "sum")
).reset_index()

# --- Output ---
print("📃 Full STA Predictions:")
print(df[["wlan_code", "x", "y", "z", "predicted_throughput", "actual_throughput"]].to_string(index=False))

print("\n📡 Predicted vs Actual AP Throughputs:")
for _, row in grouped.iterrows():
    ap_chr = chr(int(row["wlan_code"]) + ord("A")) if row["wlan_code"] >= 0 else "?"
    print(f"AP {ap_chr}: Pred = {row['predicted_ap_throughput']:.2f} Mbps | Actual = {row['actual_ap_throughput']:.2f} Mbps")

print(f"\n📊 Total Scenario Throughput: Predicted = {df['predicted_throughput'].sum():.2f} Mbps | Actual = {df['actual_throughput'].sum():.2f} Mbps")
