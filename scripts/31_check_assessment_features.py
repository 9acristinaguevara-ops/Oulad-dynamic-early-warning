
import pandas as pd
from pathlib import Path

archivo = Path(
    "data/interim/week4_assessment_features.parquet"
)

datos = pd.read_parquet(archivo)

print("TAMAÑO DEL CONJUNTO")
print("Filas:", len(datos))
print("Duplicados por matrícula:", datos.duplicated(
    subset=[
        "code_module",
        "code_presentation",
        "id_student"
    ]
).sum())

print("\nEVALUACIONES PROGRAMADAS POR MÓDULO")
print(
    datos.groupby(
        ["code_module", "code_presentation"]
    )["evaluaciones_programadas_34"]
    .first()
    .to_string()
)

print("\nESTADÍSTICAS DE LAS NUEVAS VARIABLES")
columnas = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34",
    "proporcion_entregadas_34"
]
print(datos[columnas].describe().round(2).to_string())

print("\nREVISIÓN DE VALORES IMPOSIBLES")
print(
    "Notas menores que 0:",
    (datos["promedio_notas_34"] < 0).sum()
)
print(
    "Notas mayores que 100:",
    (datos["promedio_notas_34"] > 100).sum()
)
print(
    "Aprobadas mayores que entregadas:",
    (
        datos["evaluaciones_aprobadas_34"]
        > datos["evaluaciones_entregadas_34"]
    ).sum()
)

print("\nETIQUETAS DEL OBJETIVO")
print(datos["target_future_withdrawal"].value_counts())
