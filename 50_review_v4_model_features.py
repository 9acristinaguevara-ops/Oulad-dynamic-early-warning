
import pandas as pd

df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

excluir = [
    "code_module",
    "code_presentation",
    "id_student",
    "date_registration",
    "date_unregistration",
    "final_result",
    "target_future_withdrawal"
]

print("VARIABLES DISPONIBLES PARA EL MODELO:\n")

for columna in df.columns:
    if columna not in excluir:
        print(
            columna,
            "| valores faltantes:",
            df[columna].isna().sum(),
            "| valores diferentes:",
            df[columna].nunique()
        )

print("\nVARIABLES DE EVALUACIÓN:")
print(
    df[[
        "evaluaciones_programadas_34",
        "evaluaciones_entregadas_34",
        "promedio_notas_34",
        "evaluaciones_aprobadas_34",
        "proporcion_entregadas_34",
        "entregas_tempranas_totales"
    ]].describe()
)
