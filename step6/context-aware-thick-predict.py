import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, FunctionTransformer

SCALERS_PATH = "step6/models/scalers.npz"

# --- Load data ---
X_test = np.load("step6/X_test.npy")
y_test = np.load("step6/y_test.npy")
sid_test = np.load("step6/sid_test.npy")

# --- Load scalers ---
scalers = np.load(SCALERS_PATH)
x_scaler = StandardScaler()
x_scaler.mean_ = scalers["x_mean"]
x_scaler.scale_ = scalers["x_std"]
y_transform = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)

X_test_scaled = x_scaler.transform(X_test)

# --- Define model class (same as training) ---
class ContextAwareMLP(torch.nn.Module):
    def __init__(self, input_dim=5, scenario_count=275, embed_dim=8):
        super().__init__()
        self.embed = torch.nn.Embedding(scenario_count, embed_dim)
        self.input_proj = torch.nn.Linear(input_dim + embed_dim, 128)
        self.resblocks = torch.nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.output = torch.nn.Linear(128, 1)

    def forward(self, x, sid):
        s_embed = self.embed(sid)
        x = torch.cat([x, s_embed], dim=1)
        x = self.input_proj(x)
        x = self.resblocks(x)
        return self.output(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

# --- Predict ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContextAwareMLP().to(device)
model.load_state_dict(torch.load("step6/models/context_aware_thick.pt"))
model.eval()

X_tensor = torch.from_numpy(X_test_scaled).float().to(device)
sid_tensor = torch.from_numpy(sid_test).long().to(device)

with torch.no_grad():
    preds_scaled = model(X_tensor, sid_tensor).cpu().numpy().squeeze()
    preds = y_transform.inverse_transform(preds_scaled.reshape(-1, 1)).squeeze()

# --- Output comparison ---
for i in range(20):
    print(f"GT: {y_test[i][0]:8.2f} | Pred: {preds[i]:8.2f}")
