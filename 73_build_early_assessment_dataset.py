
import pandas as pd

# Cargar el conjunto combinado actual
df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

# Seleccionar las variables de actividad
variables_actividad = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4"
]

# Seleccionar variables de participacion en evaluaciones
variables_evaluaciones = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34"
]

# Conservar identificadores, objetivo y caracteristicas
columnas = [
    "code_module",
    "code_presentation",
    "id_student",
    "target_future_withdrawal"
] + variables_actividad + variables_evaluaciones

df_nuevo = df[columnas].copy()

# Guardar una nueva version sin notas ni aprobaciones
df_nuevo.to_parquet(
    "data/interim/week4_early_assessment_v1.parquet",
    index=False
)

print("NUEVO DATASET CREADO")
print("Filas:", len(df_nuevo))
print("Columnas:", len(df_nuevo.columns))

print("\nValores faltantes:")
print(df_nuevo.isna().sum())

print("\nDistribucion del objetivo:")
print(df_nuevo["target_future_withdrawal"].value_counts())

print("\nDuplicados por estudiante y periodo:")
print(
    df_nuevo.duplicated(
        subset=[
            "code_module",
            "code_presentation",
            "id_student"
        ]
    ).sum()
)

print("\nArchivo guardado:")
print("data/interim/week4_early_assessment_v1.parquet")
