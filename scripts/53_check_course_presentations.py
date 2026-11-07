
import pandas as pd

df = pd.read_parquet(
    "data/interim/week4_combined_features_v4.parquet"
)

print("PRESENTACIONES ACADÉMICAS\n")

presentaciones = df[
    ["code_module", "code_presentation"]
].drop_duplicates()

presentaciones = presentaciones.sort_values(
    ["code_module", "code_presentation"]
)

print(presentaciones.to_string(index=False))

print("\nRESUMEN POR PRESENTACIÓN\n")

resumen = df.groupby(
    ["code_module", "code_presentation"]
).agg(
    estudiantes=("id_student", "count"),
    retiradas=("target_future_withdrawal", "sum"),
    promedio_clics=("clics_acumulados_0_4", "mean")
).reset_index()

resumen["porcentaje_retiradas"] = (
    resumen["retiradas"] / resumen["estudiantes"] * 100
)

print(resumen.round(2).to_string(index=False))

print("\nTotal de presentaciones:", len(presentaciones))
