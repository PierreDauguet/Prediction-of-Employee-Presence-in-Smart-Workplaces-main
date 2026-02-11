# ==========================================
# LSTM.py
# LSTM PyTorch pour prédire GLOBAL
# ==========================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------- Chemin robuste ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PATH = os.path.join(BASE_DIR, "data", "df_venues_processed_selected.csv")

TARGET = "GLOBAL"
LOOKBACK = 7
TEST_RATIO = 0.20
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3
SEED = 42

MODEL_OUT = os.path.join(BASE_DIR, "data", "model_lstm_torch_global.pt")


# --------- Utils ----------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# --------- Main ----------
def main():
    set_seed(SEED)

    df = pd.read_csv(PATH, sep=";")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # cast numérique
    for c in df.columns:
        if c not in {"Date", "jour_semaine"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # jour_semaine robuste
    df["jour_semaine_num"] = df["Date"].dt.dayofweek

    # features = tout sauf Date / TARGET / jour_semaine texte
    exclude = {TARGET, "Date", "jour_semaine"}
    features = [c for c in df.columns if c not in exclude]

    df = df.dropna(subset=features + [TARGET]).reset_index(drop=True)

    X = df[features].values
    y = df[TARGET].values.reshape(-1, 1)

    # standardisation
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    Xs = scaler_X.fit_transform(X)
    ys = scaler_y.fit_transform(y).flatten()

    # séquences
    X_seq, y_seq = create_sequences(Xs, ys, LOOKBACK)

    if len(X_seq) < 30:
        raise ValueError("Pas assez de données, réduis LOOKBACK")

    # split temporel
    n = len(X_seq)
    n_test = max(1, int(np.ceil(n * TEST_RATIO)))

    X_train, y_train = X_seq[:-n_test], y_seq[:-n_test]
    X_test, y_test = X_seq[-n_test:], y_seq[-n_test:]

    train_loader = DataLoader(SeqDataset(X_train, y_train),
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    test_loader = DataLoader(SeqDataset(X_test, y_test),
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(n_features=X_train.shape[2]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training...")

    for epoch in range(EPOCHS):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1} | Train MSE: {np.mean(losses):.4f}")

    # -------- Evaluation --------
    model.eval()
    preds_scaled = []
    true_scaled = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().flatten()
            preds_scaled.append(pred)
            true_scaled.append(yb.numpy().flatten())

    preds_scaled = np.concatenate(preds_scaled)
    true_scaled = np.concatenate(true_scaled)

    # inverse scaling
    y_pred = preds_scaled * scaler_y.scale_[0] + scaler_y.mean_[0]
    y_true = true_scaled * scaler_y.scale_[0] + scaler_y.mean_[0]

    mae = mean_absolute_error(y_true, y_pred)
    r = rmse(y_true, y_pred)

    print(f"\nLSTM | MAE={mae:.3f} | RMSE={r:.3f}")

    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print("✅ Modèle sauvegardé :", MODEL_OUT)


if __name__ == "__main__":
    main()
