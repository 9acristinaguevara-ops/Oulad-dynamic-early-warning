
import pandas as pd
from pathlib import Path

# 1. Rutas
ENTRADA = Path("data/interim/week4_combined_features_v4.parquet")
SALIDA = Path("outputs/tables")
SALIDA.mkdir(parents=True, exist_ok=True)

# 2. Cargar datos
df = pd.read_parquet(ENTRADA)

print("Total de estudiantes:", len(df))

# 3. Variables que vamos a estudiar
variables = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "weekly_change",
    "clics_recientes_3_4",
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34"
]

# 4. Resumen por periodo y módulo
resumen = df.groupby(
    ["code_presentation", "code_module"]
).agg(
    estudiantes=("id_student", "size"),
    retiradas=("target_future_withdrawal", "sum"),
    tasa_retirada=("target_future_withdrawal", "mean"),
    actividad_semana4=("has_week4_activity", "mean"),
    promedio_clics_semana3=("week3_clicks", "mean"),
    promedio_clics_semana4=("week4_clicks", "mean"),
    promedio_dias_activos=("week4_active_days", "mean"),
    promedio_evaluaciones=("evaluaciones_programadas_34", "mean"),
    promedio_entregas=("evaluaciones_entregadas_34", "mean")
).reset_index()

resumen["tasa_retirada"] = resumen["tasa_retirada"] * 100
resumen["actividad_semana4"] = resumen["actividad_semana4"] * 100

print("\nRESUMEN POR PERIODO Y MÓDULO")
print(resumen.round(2).to_string(index=False))

# 5. Guardar resumen
archivo_resumen = SALIDA / "91_resumen_modulo_periodo.csv"
resumen.to_csv(archivo_resumen, index=False)

# 6. Comparar promedios entre 2013B y 2013J
print("\nCOMPARACIÓN DE VARIABLES: 2013B Y 2013J")

for variable in variables:
    print("\nVariable:", variable)

    tabla = df.groupby(
        ["code_presentation", "code_module"]
    )[variable].mean().unstack("code_presentation")

    if "2013B" in tabla.columns and "2013J" in tabla.columns:
        tabla["diferencia_2013J_menos_2013B"] = (
            tabla["2013J"] - tabla["2013B"]
        )

        print(tabla.round(2).to_string())

# 7. Revisar valores faltantes por periodo
print("\nPORCENTAJE DE VALORES FALTANTES POR PERIODO")

faltantes = []

for periodo in sorted(df["code_presentation"].unique()):
    grupo = df[df["code_presentation"] == periodo]

    for variable in variables:
        faltantes.append({
            "periodo": periodo,
            "variable": variable,
            "faltantes": grupo[variable].isna().sum(),
            "porcentaje_faltante": (
                grupo[variable].isna().mean() * 100
            )
        })

tabla_faltantes = pd.DataFrame(faltantes)

print(
    tabla_faltantes.round(2).to_string(index=False)
)

archivo_faltantes = SALIDA / "91_faltantes_por_periodo.csv"
tabla_faltantes.to_csv(archivo_faltantes, index=False)

print("\nArchivos guardados:")
print(archivo_resumen)
print(archivo_faltantes)
