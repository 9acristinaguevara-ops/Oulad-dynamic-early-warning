
import pandas as pd

archivo = "data/interim/corrected_week4_dataset.parquet"

data = pd.read_parquet(archivo)

print("\nDIMENSIONES DEL CONJUNTO")
print("Filas:", data.shape[0])
print("Columnas:", data.shape[1])

print("\nNOMBRES DE LAS VARIABLES")
print(data.columns.tolist())

print("\nTIPOS DE DATOS")
print(data.dtypes)

print("\nVALORES FALTANTES")
print(data.isnull().sum())

print("\nRESUMEN DEL OBJETIVO")
print(data["target_future_withdrawal"].value_counts())

print("\nRESUMEN DE VARIABLES NUMERICAS")
print(data.describe())
