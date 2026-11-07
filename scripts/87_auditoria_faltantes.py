
import pandas as pd
from pathlib import Path


# 1. Rutas
ENTRADA = Path(
    "data/interim/week4_combined_features_v4.parquet"
)

SALIDA = Path("outputs/tables")
SALIDA.mkdir(parents=True, exist_ok=True)


# 2. Cargar datos
df = pd.read_parquet(ENTRADA)

variables = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4",
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34"
]


# 3. Faltantes por presentación
print("\n1. FALTANTES POR PRESENTACIÓN")

filas = []

for periodo in sorted(df["code_presentation"].unique()):
    grupo = df[df["code_presentation"] == periodo]

    for variable in variables:
        faltantes = grupo[variable].isna().sum()

        filas.append({
            "presentacion": periodo,
            "variable": variable,
            "estudiantes": len(grupo),
            "faltantes": faltantes,
            "porcentaje_faltante": faltantes / len(grupo) * 100
        })

faltantes_periodo = pd.DataFrame(filas)

print(
    faltantes_periodo.round(2).to_string(index=False)
)


# 4. Faltantes según el objetivo
print("\n2. FALTANTES SEGÚN EL OBJETIVO")

filas = []

for objetivo in sorted(df["target_future_withdrawal"].unique()):
    grupo = df[df["target_future_withdrawal"] == objetivo]

    for variable in variables:
        faltantes = grupo[variable].isna().sum()

        filas.append({
            "objetivo": objetivo,
            "variable": variable,
            "estudiantes": len(grupo),
            "faltantes": faltantes,
            "porcentaje_faltante": faltantes / len(grupo) * 100
        })

faltantes_objetivo = pd.DataFrame(filas)

print(
    faltantes_objetivo.round(2).to_string(index=False)
)


# 5. Comparar actividad ausente con el indicador de actividad
print("\n3. FALTANTES Y ACTIVIDAD")

for variable in [
    "week3_clicks",
    "week4_clicks",
    "week4_activity_records",
    "weekly_change",
    "clics_recientes_3_4"
]:
    print("\nVariable:", variable)

    tabla = pd.crosstab(
        df["has_week4_activity"],
        df[variable].isna(),
        normalize="index"
    ) * 100

    tabla.index.name = "has_week4_activity"
    tabla.columns = ["Disponible", "Faltante"]

    print(tabla.round(2).to_string())


# 6. Guardar resultados
faltantes_periodo.to_csv(
    SALIDA / "87_faltantes_por_presentacion.csv",
    index=False
)

faltantes_objetivo.to_csv(
    SALIDA / "87_faltantes_por_objetivo.csv",
    index=False
)

print("\nArchivos guardados:")
print("outputs/tables/87_faltantes_por_presentacion.csv")
print("outputs/tables/87_faltantes_por_objetivo.csv")
