
import pandas as pd

archivo = "data/interim/week4_combined_features.parquet"
datos = pd.read_parquet(archivo)

print("TAMANO DEL CONJUNTO")
print("Filas:", len(datos))
print("Columnas:", len(datos.columns))

print("\nVARIABLE OBJETIVO")
print("Existe target:", "target_future_withdrawal" in datos.columns)

if "target_future_withdrawal" in datos.columns:
    print(datos["target_future_withdrawal"].value_counts())

print("\nCOLUMNAS QUE NO DEBEN SER PREDICTORES")
columnas_revisar = [
    "final_result",
    "date_unregistration",
    "id_student",
    "target_future_withdrawal"
]

for columna in columnas_revisar:
    print(columna, "->", columna in datos.columns)

print("\nCOLUMNAS DISPONIBLES")
print(datos.columns.tolist())

print("\nDUPLICADOS")
print(
    datos.duplicated(
        subset=[
            "code_module",
            "code_presentation",
            "id_student"
        ]
    ).sum()
)
