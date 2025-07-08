import numpy as np

X_grouped = np.load("step6/X_grouped.npy", allow_pickle=True)
y_grouped = np.load("step6/y_grouped.npy", allow_pickle=True)

X_grouped_sta = []
y_grouped_sta = []

for X, y in zip(X_grouped, y_grouped):
    mask = (X[:, 0] == 1)  # node_type == 1 (STA)
    if mask.sum() == 0:
        continue  # skip if no stations in this scenario
    X_grouped_sta.append(X[mask])
    y_grouped_sta.append(y[mask])

print(f"✅ Scenarios with STA-only: {len(X_grouped_sta)}")

np.save("step6/X_grouped_sta.npy", np.array(X_grouped_sta, dtype=object))
np.save("step6/y_grouped_sta.npy", np.array(y_grouped_sta, dtype=object))
print("💾 Saved: X_grouped_sta.npy, y_grouped_sta.npy")
