
import pandas as pd

df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

variables = [
    "clics_acumulados_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2",
    "has_week4_activity"
]

ggg = df[df["code_module"] == "GGG"]

for periodo in ["2013J", "2014B", "2014J"]:
    grupo = ggg[ggg["code_presentation"] == periodo]

    print("\n" + "=" * 45)
    print("GGG | PERIODO:", periodo)
    print("=" * 45)

    for resultado in [0, 1]:
        estudiantes = grupo[
            grupo["target_future_withdrawal"] == resultado
        ]

        print(
            "\nResultado:",
            resultado,
            "| Estudiantes:",
            len(estudiantes)
        )

        print(
            estudiantes[variables]
            .mean()
            .round(2)
            .to_string()
        )
