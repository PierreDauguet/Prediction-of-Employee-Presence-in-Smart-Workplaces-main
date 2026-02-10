# ----------------------------
# 1) Chargement robuste + correction des noms de colonnes (FIX Ã_)
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chardet

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PATH = "data/df_venues_processed_selected.csv"

with open(PATH, "rb") as f:
    enc = chardet.detect(f.read()).get("encoding") or "utf-8"

df = pd.read_csv(PATH, sep=";", encoding=enc)

def fix_mojibake(s: str) -> str:
    """Corrige les colonnes du type '...Ã©...' -> '...é...' quand c'est possible."""
    try:
        return s.encode("latin1").decode("utf-8")
    except Exception:
        return s

def clean_col(c: str) -> str:
    """
    Corrige les 2 colonnes problématiques même si elles sont sous des variantes
    (jour_feriÃ_, jour_feri., jour_ferié, etc.), puis laisse le reste inchangé.
    """
    c = fix_mojibake(c).strip()

    # mapping "par motif" pour couvrir toutes les variantes
    if "jour" in c and "feri" in c:
        return "jour_ferie"
    if "pont" in c and "cong" in c:
        return "pont_conge"

    return c

# 1) applique la correction "intelligente"
df.columns = [clean_col(c) for c in df.columns]

# 2) nettoyage général (stable pour ML)
df.columns = (
    pd.Index(df.columns)
      .str.strip()
      .str.replace(r"[^\w]+", "_", regex=True)
)

print("Colonnes finales :")
print(df.columns)

print("\nInfos dataset :")
print(df.info())

print("\nStatistiques :")
print(df.describe())


# ----------------------------
# 2) Données numériques + suppression des colonnes constantes
# ----------------------------
df_num = df.select_dtypes(include=[np.number]).copy()

# Supprime les colonnes constantes (corr = NaN sinon)
nunique = df_num.nunique(dropna=False)
const_cols = nunique[nunique <= 1].index.tolist()
if const_cols:
    print("\nColonnes constantes supprimées (corrélation NaN sinon) :", const_cols)
    df_num = df_num.drop(columns=const_cols)


# ----------------------------
# 3) Corrélations + heatmap + corrélation avec une cible
# ----------------------------
corr = df_num.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=False, center=0)
plt.title("Matrice de corrélation (numérique)")
plt.tight_layout()
plt.show()

# Choix de la cible (GLOBAL si dispo)
TARGET = "GLOBAL" if "GLOBAL" in df_num.columns else df_num.columns[0]
corr_target = corr[TARGET].sort_values(ascending=False)

print(f"\nCorrélation avec {TARGET} :")
print(corr_target)

plt.figure(figsize=(8, 6))
corr_target.drop(index=TARGET).plot(kind="bar")
plt.title(f"Corrélation avec {TARGET}")
plt.tight_layout()
plt.show()


# ----------------------------
# 4) Graphes "conventionnels" : distributions + boxplots
# ----------------------------
# Histogrammes
df_num.hist(figsize=(12, 10), bins=30)
plt.suptitle("Histogrammes des variables numériques")
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_num)
plt.xticks(rotation=90)
plt.title("Boxplots des variables numériques")
plt.tight_layout()
plt.show()


# ----------------------------
# 5) ACP (PCA) : standardisation + scree plot + individus + cercle corrélations
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num.values)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Scree plot (variance expliquée cumulée)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.title("ACP - Variance expliquée cumulée (Scree plot)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Projection des individus (PC1 vs PC2)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("ACP - Projection des individus (PC1 vs PC2)")
plt.axhline(0)
plt.axvline(0)
plt.grid(True)
plt.tight_layout()
plt.show()

# Cercle des corrélations (chargements)
# On normalise les loadings pour les afficher dans le cercle unité
loadings = pca.components_.T  # shape: (n_features, n_components)
features = df_num.columns.tolist()

plt.figure(figsize=(8, 8))
for i, feat in enumerate(features):
    x, y = loadings[i, 0], loadings[i, 1]
    plt.arrow(0, 0, x, y, head_width=0.03, length_includes_head=True)
    plt.text(x * 1.08, y * 1.08, feat, fontsize=9)

circle = plt.Circle((0, 0), 1, fill=False)
plt.gca().add_artist(circle)

plt.xlabel("PC1 (loadings)")
plt.ylabel("PC2 (loadings)")
plt.title("ACP - Cercle des corrélations (PC1 vs PC2)")
plt.axhline(0)
plt.axvline(0)
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()

test = df["GLOBAL"] - (df["D1"] + df["D2"] + df["D3"] + df["D4"])
print(test.abs().max())

# ----------------------------
# 6) (Optionnel) Export CSV propre en UTF-8 (pour ne plus jamais avoir le souci)
# ----------------------------
OUT = "data/df_venues_processed_clean.csv"
df.to_csv(OUT, sep=";", index=False, encoding="utf-8-sig")
print(f"\nCSV propre exporté : {OUT}")
