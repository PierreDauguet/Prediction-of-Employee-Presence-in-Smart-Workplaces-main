# ==========================================
# RobFTS.py
# Version discrétisée robuste du modèle RobFTS
# ==========================================

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import HuberRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib


# --------- Chemin robuste ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "df_venues_processed_selected.csv")
MODEL_OUT = os.path.join(BASE_DIR, "data", "model_robfts_discret_huber.joblib")

TARGET_COL = "GLOBAL"

# Colonnes exogènes (si présentes dans le CSV)
EXOG_COLS = ["Temp", "pluie", "Total_reservations"]

L = 2            # nombre de semaines passées utilisées
TEST_WEEKS = 8   # nb de semaines en test
DAYS = [0, 1, 2, 3, 4]  # lundi-vendredi


# --------- Utils ----------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_week_curves(df, value_cols):
    """
    Construit une table avec une ligne = une semaine,
    et chaque colonne contient un vecteur (5 jours).
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce", dayfirst=True)
    d = d.dropna(subset=["Date"]).sort_values("Date")

    d["dow"] = d["Date"].dt.dayofweek
    d = d[d["dow"].isin(DAYS)]

    iso = d["Date"].dt.isocalendar()
    d["year"] = iso.year.astype(int)
    d["week"] = iso.week.astype(int)
    d["yw"] = d["year"].astype(str) + "-" + d["week"].astype(str).str.zfill(2)

    rows = []

    for yw, g in d.groupby("yw"):
        g = g.sort_values("dow")

        if len(g) != len(DAYS):
            continue

        row = {"yw": yw}

        for col in value_cols:
            if col not in g.columns:
                continue

            vals = pd.to_numeric(g[col], errors="coerce").values
            if np.any(pd.isna(vals)):
                row[col] = None
            else:
                row[col] = vals.astype(float)

        rows.append(row)

    W = pd.DataFrame(rows).set_index("yw")

    for col in value_cols:
        if col in W.columns:
            W = W[W[col].notna()]

    return W


def flatten_curve(vec):
    return vec.reshape(-1)


# --------- Main ----------
def main():

    df = pd.read_csv(CSV_PATH, sep=";")

    used_cols = [TARGET_COL] + [c for c in EXOG_COLS if c in df.columns]

    W = build_week_curves(df, used_cols)

    weeks = W.index.tolist()
    W = W.loc[weeks].copy()

    X_list = []
    Y_list = []

    for i in range(L, len(W) - 1):

        y_next = W.iloc[i + 1][TARGET_COL]

        feats = []

        # L lags fonctionnels
        for lag in range(L):
            feats.append(flatten_curve(W.iloc[i - lag][TARGET_COL]))

        # exogènes de la semaine i
        for col in EXOG_COLS:
            if col in W.columns:
                feats.append(flatten_curve(W.iloc[i][col]))

        x = np.concatenate(feats)
        X_list.append(x)
        Y_list.append(flatten_curve(y_next))

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    if len(X) < (TEST_WEEKS + 5):
        raise ValueError("Pas assez de semaines. Réduis TEST_WEEKS ou L.")

    # Split temporel
    X_train, X_test = X[:-TEST_WEEKS], X[-TEST_WEEKS:]
    Y_train, Y_test = Y[:-TEST_WEEKS], Y[-TEST_WEEKS:]

    # Pipeline robuste + scaling
    base_model = make_pipeline(
        StandardScaler(),
        HuberRegressor(
            epsilon=1.35,
            alpha=1e-4,
            max_iter=5000
        )
    )

    model = MultiOutputRegressor(base_model)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    mae = mean_absolute_error(Y_test, Y_pred)
    r = rmse(Y_test, Y_pred)

    print(f"RobFTS-discret (Huber) | MAE={mae:.3f} | RMSE={r:.3f} | Train={len(X_train)} Test={len(X_test)}")

    print("\nExemple prédiction (5 jours) :", np.round(Y_pred[-1], 1))
    print("Réel (5 jours)               :", np.round(Y_test[-1], 1))

    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    joblib.dump(model, MODEL_OUT)

    print("\n✅ Modèle sauvegardé :", MODEL_OUT)


if __name__ == "__main__":
    main()
