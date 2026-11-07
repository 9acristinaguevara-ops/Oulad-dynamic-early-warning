
import pandas as pd

df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

# Orden cronológico de los periodos académicos
orden_periodos = [
    "2013B",
    "2013J",
    "2014B",
    "2014J"
]

print("RESUMEN POR PERIODO ACADÉMICO\n")

resumen = df.groupby(
    "code_presentation"
).agg(
    estudiantes=("id_student", "count"),
    retiradas=("target_future_withdrawal", "sum")
).reset_index()

resumen["porcentaje_retiradas"] = (
    resumen["retiradas"] / resumen["estudiantes"] * 100
)

resumen["code_presentation"] = pd.Categorical(
    resumen["code_presentation"],
    categories=orden_periodos,
    ordered=True
)

resumen = resumen.sort_values("code_presentation")

print(resumen.round(2).to_string(index=False))

print("\nPROPUESTA DE SEPARACIÓN TEMPORAL")
print("Entrenamiento: 2013B y 2013J")
print("Prueba: 2014B y 2014J")

print("\nEstudiantes por periodo:")
for periodo in orden_periodos:
    cantidad = (
        df["code_presentation"] == periodo
    ).sum()
    print(periodo, ":", cantidad)
