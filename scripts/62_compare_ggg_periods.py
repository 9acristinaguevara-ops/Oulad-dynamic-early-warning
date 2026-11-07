
import pandas as pd

df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

variables = [
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2",
    "has_week4_activity"
]

ggg = df[df["code_module"] == "GGG"]

print("COMPARACION DE GGG POR PERIODO\n")

for periodo in ["2013J", "2014B", "2014J"]:
    grupo = ggg[ggg["code_presentation"] == periodo]

    print("\nPeriodo:", periodo)
    print("Estudiantes:", len(grupo))

    print(
        "Tasa de retiradas:",
        round(
            grupo["target_future_withdrawal"].mean() * 100,
            2
        ),
        "%"
    )

    print("\nPromedios:")
    print(grupo[variables].mean().round(2).to_string())

    print("\nMedianas:")
    print(grupo[variables].median().round(2).to_string())
