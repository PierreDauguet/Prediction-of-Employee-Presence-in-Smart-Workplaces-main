# ==========================================
# train_nlp_global.py
# NLP (TF-IDF) -> prédiction de GLOBAL
# ==========================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


# ---- chemins robustes (si script dans /model) ----
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "data", "df_venues_processed_selected.csv")

TARGET = "GLOBAL"
TEST_RATIO = 0.20

MODEL_OUT = os.path.join(BASE_DIR, "data", "model_nlp_tfidf_ridge.joblib")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def row_to_text(row: pd.Series) -> str:
    """
    Convertit une ligne en "texte" pour NLP.
    On discrétise les variables numériques (bins) pour créer des tokens stables.
    """
    tokens = []

    # Jour de semaine (si texte, on garde; sinon on dérive de Date)
    if "jour_semaine" in row.index and pd.notna(row["jour_semaine"]):
        tokens.append(f"jour={str(row['jour_semaine']).lower()}")
    elif "Date" in row.index and pd.notna(row["Date"]):
        # optionnel
        pass

    # Semaine_sin déjà dans ton CSV (token continu -> on le met en bins)
    if "Semaine_sin" in row.index and pd.notna(row["Semaine_sin"]):
        v = float(row["Semaine_sin"])
        tokens.append(f"semaine_bin={int(np.floor((v + 1) * 5))}")  # 0..10

    # Binaires / événements (0/1)
    for c in ["jour_ferie", "pont_conge", "holiday", "autre", "Greve_nationale", "prof_nationale"]:
        if c in row.index and pd.notna(row[c]):
            tokens.append(f"{c}={int(float(row[c]))}")

    # Numériques -> bins (pour en faire des "mots")
    def bin_token(name, value, bins):
        try:
            v = float(value)
        except Exception:
            return
        b = np.digitize([v], bins)[0]  # 1..len(bins)+1
        tokens.append(f"{name}_bin={b}")

    if "Temp" in row.index and pd.notna(row["Temp"]):
        bin_token("temp", row["Temp"], bins=[0, 5, 10, 15, 20, 25, 30, 35])

    if "pluie" in row.index and pd.notna(row["pluie"]):
        bin_token("pluie", row["pluie"], bins=[0, 0.1, 0.5, 1, 2, 5, 10, 20])

    if "Total_reservations" in row.index and pd.notna(row["Total_reservations"]):
        bin_token("resa", row["Total_reservations"], bins=[0, 20, 50, 100, 150, 200, 250, 300, 350, 500])

    # (Optionnel) Date en token de mois
    if "Date" in row.index and pd.notna(row["Date"]):
        try:
            dt = pd.to_datetime(row["Date"], errors="coerce", dayfirst=True)
            if pd.notna(dt):
                tokens.append(f"mois={dt.month}")
        except Exception:
            pass

    return " ".join(tokens)


def main():
    df = pd.read_csv(CSV_PATH, sep=";")

    # parsing + tri temporel
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # cible
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # texte NLP
    df["text"] = df.apply(row_to_text, axis=1)

    # split temporel (dernier TEST_RATIO en test)
    n = len(df)
    n_test = max(1, int(np.ceil(n * TEST_RATIO)))
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()

    X_train, y_train = train_df["text"].values, train_df[TARGET].values
    X_test, y_test = test_df["text"].values, test_df[TARGET].values

    # pipeline NLP : TF-IDF -> Ridge
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("ridge", Ridge(alpha=5.0, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    r = rmse(y_test, y_pred)
    print(f"NLP TF-IDF + Ridge | MAE={mae:.3f} | RMSE={r:.3f} | Train={len(train_df)} Test={len(test_df)}")

    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"✅ Modèle sauvegardé : {MODEL_OUT}")

    print("\nExemple texte (dernière ligne test) :")
    print(test_df["text"].iloc[-1])
    print(f"GLOBAL réel={y_test[-1]} | GLOBAL prédite={y_pred[-1]:.2f}")


if __name__ == "__main__":
    main()
