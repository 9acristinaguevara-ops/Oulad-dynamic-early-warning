
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
    "evaluaciones_entregadas_34",
    "proporcion_entregadas_34",
    "evaluaciones_aprobadas_34",
    "has_week4_activity"
]

df = df[
    df["code_presentation"].isin(["2014B", "2014J"]) &
    df["code_module"].isin(["CCC", "GGG"])
]

for modulo in ["CCC", "GGG"]:
    for periodo in ["2014B", "2014J"]:
        grupo = df[
            (df["code_module"] == modulo) &
            (df["code_presentation"] == periodo)
        ]

        print("\n" + "=" * 55)
        print("MODULO:", modulo, "| PERIODO:", periodo)
        print("=" * 55)

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
