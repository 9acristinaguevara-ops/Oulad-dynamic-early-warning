
import pandas as pd

df = pd.read_parquet(
    "data/interim/week4_combined_features.parquet"
)

print("REVISIÓN DE PREDICTORES")
print("------------------------------")

print("Filas:", len(df))
print("Columnas:", len(df.columns))

print("\nVARIABLES DE EVALUACIONES")
print("------------------------------")

columnas = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34",
    "proporcion_entregadas_34",
    "entregas_tempranas_totales"
]

for columna in columnas:
    if columna in df.columns:
        print("\nVariable:", columna)
        print("Valores faltantes:", df[columna].isna().sum())
        print("Valores únicos:", df[columna].nunique())
        print("Mínimo:", df[columna].min())
        print("Máximo:", df[columna].max())

print("\nVARIABLES DE ACTIVIDAD")
print("------------------------------")

columnas_actividad = [
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "promedio_clics_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2"
]

for columna in columnas_actividad:
    if columna in df.columns:
        print(
            columna,
            "| faltantes:", df[columna].isna().sum(),
            "| mínimo:", df[columna].min(),
            "| máximo:", df[columna].max()
        )
