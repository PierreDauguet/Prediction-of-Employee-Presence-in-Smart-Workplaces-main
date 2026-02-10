# ==========================================
# prepare_dataset.py
# Sélection par 0/1 + ajout Semaine_sin/cos + génération CSV + appel script principal
# ==========================================

import pandas as pd
import numpy as np
import chardet
import subprocess

IN_PATH = "data/df_venues_processed.csv"
OUT_PATH = "data/df_venues_processed_selected.csv"

# ----------------------------------
# 1) Détection encodage + lecture
# ----------------------------------
with open(IN_PATH, "rb") as f:
    enc = chardet.detect(f.read()).get("encoding") or "utf-8"

df = pd.read_csv(IN_PATH, sep=";", encoding=enc)

# ----------------------------------
# 2) Choix des colonnes (1 = garder, 0 = supprimer)
# 👉 MODIFIE ICI
# ----------------------------------
COLUMN_SELECTION = {
    "Date": 1,
    "GLOBAL": 1,
    "D1": 0,
    "D2": 0,
    "D3": 0,
    "D4": 0,
    "jour_ferie": 1,
    "pont_conge": 1,
    "holiday": 1,
    "jour_semaine": 1,
    "Semaine": 1,            # <- si 1, on la transforme en sin/cos
    "Annee": 0,
    "Annee_et_Semaine": 0,
    "Temp": 1,
    "pluie": 1,
    "autre": 1,
    "Greve_nationale": 1,
    "SNCF": 0,
    "prof_nationale": 1,
    "Total_reservations": 1
}

# ----------------------------------
# 3) Application sélection
# ----------------------------------
selected_cols = [
    col for col, keep in COLUMN_SELECTION.items()
    if keep == 1 and col in df.columns
]

missing_cols = [col for col in COLUMN_SELECTION.keys() if col not in df.columns]
if missing_cols:
    print("⚠ Colonnes absentes :", missing_cols)

df_filtered = df[selected_cols].copy()

# ----------------------------------
# 4) Transformation cyclique de Semaine -> sin/cos
# ----------------------------------
# On ne fait la transfo que si :
# - la colonne Semaine est gardée
# - elle existe bien
if "Semaine" in df_filtered.columns:
    # conversion numérique robuste
    df_filtered["Semaine"] = pd.to_numeric(df_filtered["Semaine"], errors="coerce")

    # 52 semaines dans une année (standard)
    period = 52

    df_filtered["Semaine_sin"] = np.sin(2 * np.pi * df_filtered["Semaine"] / period)

    # on supprime la colonne originale (souvent préférable)
    df_filtered = df_filtered.drop(columns=["Semaine"])

print("\nColonnes conservées (après transformation semaine) :")
print(df_filtered.columns.tolist())

# ----------------------------------
# 5) Export CSV propre
# ----------------------------------
df_filtered.to_csv(OUT_PATH, sep=";", index=False, encoding="utf-8-sig")
print(f"\n✅ CSV filtré généré : {OUT_PATH}")

# ----------------------------------
# 6) Lancement automatique du script principal
# ----------------------------------
MAIN_SCRIPT = "analyse_data.py"   # <-- nom de ton autre script

print(f"\n🚀 Lancement de {MAIN_SCRIPT} ...\n")
subprocess.run(["python", MAIN_SCRIPT])
