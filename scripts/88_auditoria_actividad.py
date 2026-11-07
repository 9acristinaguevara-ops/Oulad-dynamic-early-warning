
import pandas as pd
from pathlib import Path


# 1. Cargar datos
archivo = Path("data/interim/week4_combined_features_v4.parquet")
salida = Path("outputs/tables")
salida.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(archivo)


# 2. Revisar valores faltantes
print("\n1. FALTANTES DE ACTIVIDAD")

variables = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "weekly_change",
    "clics_recientes_3_4"
]

print(df[variables].isna().sum().to_string())


# 3. Comprobar la semana 4
print("\n2. CONSISTENCIA DE SEMANA 4")

sin_actividad = df["has_week4_activity"] == 0
con_actividad = df["has_week4_activity"] == 1

print("Estudiantes sin actividad semana 4:", sin_actividad.sum())
print("Estudiantes con actividad semana 4:", con_actividad.sum())

for variable in [
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records"
]:
    print("\nVariable:", variable)

    print(
        "Sin actividad, valor faltante:",
        df.loc[sin_actividad, variable].isna().sum()
    )

    print(
        "Sin actividad, valor disponible:",
        df.loc[sin_actividad, variable].notna().sum()
    )

    print(
        "Con actividad, valor faltante:",
        df.loc[con_actividad, variable].isna().sum()
    )

    print(
        "Con actividad, valor disponible:",
        df.loc[con_actividad, variable].notna().sum()
    )


# 4. Comprobar la semana 3
print("\n3. CONSISTENCIA DE SEMANA 3")

week3_faltante = df["week3_clicks"].isna()

print("Semana 3 sin datos:", week3_faltante.sum())
print(
    "Semana 3 sin datos, pero con actividad en semana 4:",
    (week3_faltante & con_actividad).sum()
)

print(
    "Semana 3 sin datos y sin actividad en semana 4:",
    (week3_faltante & sin_actividad).sum()
)


# 5. Comprobar cambio semanal
print("\n4. CONSISTENCIA DE WEEKLY_CHANGE")

esperado = df["week4_clicks"] - df["week3_clicks"]
ambos_disponibles = (
    df["week4_clicks"].notna()
    & df["week3_clicks"].notna()
)

diferencia = (
    df.loc[ambos_disponibles, "weekly_change"]
    - esperado[ambos_disponibles]
).abs()

print("Filas con ambas semanas disponibles:", ambos_disponibles.sum())
print(
    "Filas donde weekly_change falta aunque ambas semanas están disponibles:",
    df.loc[ambos_disponibles, "weekly_change"].isna().sum()
)
print(
    "Diferencias distintas de cero:",
    (diferencia > 0.001).sum()
)


# 6. Comparar los valores faltantes por presentación
print("\n5. FALTANTES POR PRESENTACIÓN")

resumen = df.groupby("code_presentation").agg(
    estudiantes=("id_student", "size"),
    sin_actividad_semana4=("has_week4_activity", lambda x: (x == 0).sum()),
    faltantes_semana3=("week3_clicks", lambda x: x.isna().sum()),
    faltantes_semana4=("week4_clicks", lambda x: x.isna().sum())
)

resumen["pct_sin_actividad_semana4"] = (
    resumen["sin_actividad_semana4"] / resumen["estudiantes"] * 100
)

resumen["pct_faltantes_semana3"] = (
    resumen["faltantes_semana3"] / resumen["estudiantes"] * 100
)

resumen["pct_faltantes_semana4"] = (
    resumen["faltantes_semana4"] / resumen["estudiantes"] * 100
)

print(resumen.round(2).to_string())


# 7. Guardar resultados
resumen.to_csv(salida / "88_auditoria_actividad_por_periodo.csv")

print("\nArchivo guardado:")
print("outputs/tables/88_auditoria_actividad_por_periodo.csv")
