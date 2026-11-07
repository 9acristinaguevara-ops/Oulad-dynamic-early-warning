
import pandas as pd

df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

variables = [
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "promedio_clics_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2",
    "evaluaciones_entregadas_34",
    "proporcion_entregadas_34",
    "evaluaciones_aprobadas_34"
]

print("PROMEDIOS DE VARIABLES POR PERIODO\n")

resumen = df.groupby("code_presentation")[variables].mean()

print(resumen.round(2).to_string())

print("\nPORCENTAJE DE VALORES FALTANTES POR PERIODO\n")

faltantes = df.groupby("code_presentation")[variables].apply(
    lambda grupo: grupo.isna().mean() * 100
)

print(faltantes.round(2).to_string())

print("\nTASA DE RETIRADAS POR PERIODO\n")

tasas = df.groupby("code_presentation")[
    "target_future_withdrawal"
].mean() * 100

print(tasas.round(2).to_string())
