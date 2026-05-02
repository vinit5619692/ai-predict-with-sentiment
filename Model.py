"""
model.py
LSTM model — trained on macro factors only.
Predicts next-day direction (up/down) for TATASTEEL.NS
"""

import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from config import DB_PATH

# ── Device — NVIDIA CUDA if available, else CPU
# Intel iGPU is NOT supported by PyTorch CUDA
torch.set_num_threads(4)  # use 4 CPU threads when on CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[model] Using device: {DEVICE}")

# ── Config ──────────────────────────────
SEQ_LEN    = 10
BATCH_SIZE = 32
EPOCHS     = 150
LR         = 1e-3
HIDDEN     = 64
LAYERS     = 2

FEATURE_COLS = [
    "crude_usd", "crude_chg_pct", "crude_lag3", "crude_lag7",
    "inr_usd",   "inr_chg_pct",   "inr_lag2",
    "fii_net_crore", "fii_3d_avg", "fii_lag1",
    "fii_ls_ratio",  "fii_ls_5d_avg", "fii_ls_lag2",
    "fii_net_futures_value",
    "geo_score", "geo_5d_avg", "geo_lag5",
    "policy_score", "policy_5d_avg", "policy_lag10",
    "rbi_repo_rate", "repo_chg", "repo_lag30",
]


# ── Model ────────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()   # output: probability of UP
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


# ── Data prep ────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM features", conn, index_col="index")
    conn.close()
    df.dropna(inplace=True)

    X = df[FEATURE_COLS].values
    y = df["target_direction"].values

    X = pd.DataFrame(X).replace([float("inf"), float("-inf")], pd.NA).ffill().fillna(0).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Xs, ys = [], []
    for i in range(SEQ_LEN, len(X)):
        Xs.append(X[i-SEQ_LEN:i])
        ys.append(y[i])

    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    split = int(0.8 * len(Xs))
    return (Xs[:split], ys[:split], Xs[split:], ys[split:], scaler)


# ── Train ────────────────────────────────
def train():
    X_train, y_train, X_val, y_val, scaler = load_data()

    # pin_memory only useful with CUDA; skip on CPU
    pin = DEVICE.type == "cuda"
    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=pin, num_workers=0
    )

    model = StockLSTM(input_size=len(FEATURE_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val).to(DEVICE)
                y_val_t = torch.tensor(y_val).to(DEVICE)
                val_pred = model(X_val_t)
                val_acc  = ((val_pred > 0.5).float() == y_val_t).float().mean()
            print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_dl):.4f} | Val Acc: {val_acc:.3f}")

    torch.save({"model": model.state_dict(), "scaler": scaler,
                "feature_cols": FEATURE_COLS, "seq_len": SEQ_LEN},
               "tatasteel_model.pt")
    print("Model saved: tatasteel_model.pt")
    return model, scaler


# ── Predict latest ───────────────────────
def predict_latest():
    checkpoint = torch.load("tatasteel_model.pt", weights_only=False)
    scaler     = checkpoint["scaler"]
    feat_cols  = checkpoint["feature_cols"]
    seq_len    = checkpoint["seq_len"]

    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM features", conn, index_col="index")
    conn.close()
    df.dropna(inplace=True)

    X = scaler.transform(df[feat_cols].values)
    seq = torch.tensor(X[-seq_len:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model = StockLSTM(input_size=len(feat_cols)).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        prob_up = model(seq).item()

    direction = "UP 📈" if prob_up > 0.5 else "DOWN 📉"
    print(f"\nTATASTEEL.NS prediction for tomorrow:")
    print(f"  Direction : {direction}")
    print(f"  Confidence: {max(prob_up, 1-prob_up)*100:.1f}%")


if __name__ == "__main__":
    train()
    predict_latest()