
import pandas as pd
from pathlib import Path

archivo = Path("data/interim/week4_enriched_features.parquet")
datos = pd.read_parquet(archivo)

columnas = [
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "promedio_clics_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2",
    "target_future_withdrawal"
]

print("TAMAÑO DEL CONJUNTO")
print("Filas:", len(datos))
print("Columnas:", len(datos.columns))

print("\nESTADÍSTICAS DE LAS VARIABLES")
print(datos[columnas].describe().round(2))

print("\nVALORES FALTANTES")
print(datos[columnas].isna().sum())

print("\nDISTRIBUCIÓN DEL OBJETIVO")
print(datos["target_future_withdrawal"].value_counts())

print("\nVERIFICACIÓN DE VALORES NEGATIVOS")
print((datos[[
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "dias_activos_0_4"
]] < 0).sum())
