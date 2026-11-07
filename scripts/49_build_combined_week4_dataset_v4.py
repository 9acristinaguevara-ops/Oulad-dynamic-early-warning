
import pandas as pd

actividad = pd.read_parquet(
    "data/interim/week4_enriched_features.parquet"
)

evaluaciones = pd.read_parquet(
    "data/interim/week4_assessment_features_v4.parquet"
)

columnas_evaluacion = [
    "code_module",
    "code_presentation",
    "id_student",
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34",
    "proporcion_entregadas_34",
    "entregas_tempranas_totales"
]

evaluaciones = evaluaciones[columnas_evaluacion]

df = pd.merge(
    actividad,
    evaluaciones,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
    validate="one_to_one"
)

df.to_parquet(
    "data/interim/week4_combined_features_v4.parquet",
    index=False
)

print("Dataset combinado V4 creado correctamente")
print("Filas:", len(df))
print("Columnas:", len(df.columns))

print(
    "Duplicados por matrícula:",
    df.duplicated(
        subset=["code_module", "code_presentation", "id_student"]
    ).sum()
)

print("\nDistribución del objetivo:")
print(df["target_future_withdrawal"].value_counts())

print("\nValores faltantes en variables de evaluación:")
print(
    df[[
        "evaluaciones_programadas_34",
        "evaluaciones_entregadas_34",
        "promedio_notas_34",
        "evaluaciones_aprobadas_34",
        "proporcion_entregadas_34",
        "entregas_tempranas_totales"
    ]].isna().sum()
)
