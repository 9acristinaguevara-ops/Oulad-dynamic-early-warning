
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

archivo = (
    BASE / "data" / "interim" /
    "week4_combined_features_v4.parquet"
)

df = pd.read_parquet(archivo)

print("COLUMNAS DEL ARCHIVO:")
print(df.columns.tolist())

print("\nPRIMERAS CINCO FILAS:")
print(df.head())

print("\nPOSIBLES VARIABLES DE RESULTADO:")

for columna in df.columns:
    if any(
        palabra in columna.lower()
        for palabra in [
            "target",
            "withdraw",
            "aband",
            "result",
            "retiro",
            "final"
        ]
    ):
        print("\nColumna:", columna)
        print(df[columna].value_counts(dropna=False).head(10))
