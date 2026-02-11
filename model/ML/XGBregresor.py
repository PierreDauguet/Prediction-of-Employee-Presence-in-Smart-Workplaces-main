# ==========================================
# train_xgboost_global.py
# Entraîne un XGBoost Regressor pour prédire GLOBAL
# ==========================================

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error

# pip install xgboost
from xgboost import XGBRegressor


PATH = "../data/df_venues_processed_selected.csv"
TARGET = "GLOBAL"

LOOKBACK = 7          # 3/5/7/14 à tester selon tes lignes
TEST_RATIO = 0.20
RANDOM_STATE = 42

MODEL_OUT = "../data/model_xgb_global.joblib"
FEATURES_OUT = "../data/model_xgb_features.txt"


def add_lags(df: pd.DataFrame, col: str, lookback: int) -> pd.DataFrame:
    df = df.copy()
    for lag in range(1, lookback + 1):
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    df = pd.read_csv(PATH, sep=";")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # cast numérique robuste (sauf Date / jour_semaine texte)
    for c in df.columns:
        if c not in {"Date", "jour_semaine"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # jour_semaine -> version numérique robuste
    if "jour_semaine" in df.columns and df["jour_semaine"].dtype == "object":
        df["jour_semaine_num"] = df["Date"].dt.dayofweek
    elif "jour_semaine" in df.columns:
        df["jour_semaine_num"] = pd.to_numeric(df["jour_semaine"], errors="coerce")
    else:
        df["jour_semaine_num"] = df["Date"].dt.dayofweek

    # lags de la cible
    df = add_lags(df, TARGET, LOOKBACK)

    # features = toutes les colonnes sauf Date, TARGET, jour_semaine (texte)
    exclude = {TARGET, "Date", "jour_semaine"}
    features = [c for c in df.columns if c not in exclude]

    # lignes complètes
    df_model = df.dropna(subset=features + [TARGET]).reset_index(drop=True)
    if len(df_model) < 30:
        raise ValueError(
            f"Pas assez de lignes utilisables ({len(df_model)}). "
            f"Réduis LOOKBACK (ex: 3/5/7) ou vérifie Date / NaN."
        )

    # split temporel
    n = len(df_model)
    n_test = max(1, int(np.ceil(n * TEST_RATIO)))
    train_df = df_model.iloc[:-n_test].copy()
    test_df = df_model.iloc[-n_test:].copy()

    X_train = train_df[features].values
    y_train = train_df[TARGET].values
    X_test = test_df[features].values
    y_test = test_df[TARGET].values

    # modèle XGBoost
    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train)

    # évaluation
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r = rmse(y_test, y_pred)
    print(f"XGBoost | MAE={mae:.3f} | RMSE={r:.3f} | Train={len(train_df)} Test={len(test_df)}")

    # sauvegarde
    joblib.dump(model, MODEL_OUT)
    with open(FEATURES_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(features))

    print(f"✅ Modèle sauvegardé : {MODEL_OUT}")
    print(f"✅ Liste features     : {FEATURES_OUT}")

    # exemple : prédire la dernière ligne utilisable
    last_row = df_model.iloc[[-1]][features].values
    pred_last = float(model.predict(last_row)[0])
    print(f"Exemple prédiction (dernière ligne utilisable) -> GLOBAL prédite = {pred_last:.2f}")


if __name__ == "__main__":
    main()
