
import pandas as pd

archivo = "data/interim/week4_assessment_features_v2.parquet"
datos = pd.read_parquet(archivo)

print("TAMAÑO DEL CONJUNTO")
print("Filas:", len(datos))

print("\nDUPLICADOS POR MATRÍCULA")
print(datos.duplicated(
    subset=[
        "code_module",
        "code_presentation",
        "id_student"
    ]
).sum())

print("\nESTADÍSTICAS DE EVALUACIONES")
columnas = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34",
    "proporcion_entregadas_34"
]

print(datos[columnas].describe().round(2).to_string())

print("\nESTUDIANTES SIN ENTREGAS TEMPRANAS")
print(
    (datos["evaluaciones_entregadas_34"] == 0).sum()
)

print("\nNOTAS FUERA DEL RANGO 0-100")
print(
    (
        (datos["promedio_notas_34"] < 0)
        | (datos["promedio_notas_34"] > 100)
    ).sum()
)

print("\nETIQUETAS")
print(datos["target_future_withdrawal"].value_counts())

print("\nVALORES FALTANTES")
print(datos[columnas].isna().sum().to_string())
