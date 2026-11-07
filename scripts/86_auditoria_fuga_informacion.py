
import pandas as pd
from pathlib import Path


# 1. Rutas
BASE = Path(".")
ARCHIVO = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"

SALIDA.mkdir(parents=True, exist_ok=True)


# 2. Cargar datos
df = pd.read_parquet(ARCHIVO)

print("\nAUDITORIA DE FUGA DE INFORMACION")
print("Filas:", len(df))


# 3. Revisar identificadores y duplicados
claves = ["code_module", "code_presentation", "id_student"]

duplicados = df.duplicated(subset=claves).sum()

print("\n1. IDENTIFICADORES")
print("Filas duplicadas por estudiante y presentación:", duplicados)
print("Identificadores vacíos:")
print(df[claves].isna().sum().to_string())


# 4. Revisar objetivo
print("\n2. OBJETIVO")
print(df["target_future_withdrawal"].value_counts(dropna=False).to_string())

print("\nTasa de abandono por presentación:")
tasas = df.groupby("code_presentation")[
    "target_future_withdrawal"
].agg(["count", "sum", "mean"])

tasas["mean"] = tasas["mean"] * 100
print(tasas.round(2).to_string())


# 5. Revisar variables utilizadas por el modelo
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

print("\n3. VALORES FALTANTES")
print(df[variables].isna().sum().to_string())

print("\n4. VALORES MINIMOS")
print(df[variables].min().round(3).to_string())

print("\n5. VALORES MAXIMOS")
print(df[variables].max().round(3).to_string())


# 6. Revisar si las variables están asociadas con el objetivo
print("\n6. PROMEDIOS SEGUN EL OBJETIVO")

promedios = df.groupby("target_future_withdrawal")[variables].mean().T
promedios.columns = ["No abandono", "Abandono"]

print(promedios.round(3).to_string())


# 7. Revisar variables de cambio y consistencia
print("\n7. COMPROBACIONES DE CONSISTENCIA")

if "week3_clicks" in df.columns and "week4_clicks" in df.columns:
    cambio_calculado = df["week4_clicks"] - df["week3_clicks"]

    if "weekly_change" in df.columns:
        diferencia = (
            df["weekly_change"] - cambio_calculado
        ).abs()

        print(
            "Filas donde weekly_change no coincide con "
            "week4_clicks - week3_clicks:",
            (diferencia > 0.001).sum()
        )

if "has_week4_activity" in df.columns:
    filas_sin_actividad = (
        (df["has_week4_activity"] == 0)
        & (df["week4_activity_records"] > 0)
    ).sum()

    print(
        "Filas marcadas sin actividad pero con registros:",
        filas_sin_actividad
    )


# 8. Guardar los resúmenes
tasas.to_csv(SALIDA / "86_tasas_por_presentacion.csv")
promedios.to_csv(SALIDA / "86_promedios_por_objetivo.csv")

print("\nArchivos guardados:")
print("outputs/tables/86_tasas_por_presentacion.csv")
print("outputs/tables/86_promedios_por_objetivo.csv")

print(
    "\nIMPORTANTE: este script realiza comprobaciones iniciales. "
    "No demuestra por sí solo que exista o no fuga de información."
)
