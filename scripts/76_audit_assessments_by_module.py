
import pandas as pd

# 1. Cargar los datos
df = pd.read_parquet(
    "data/interim/week4_early_assessment_v1.parquet"
)

# 2. Revisar las variables de evaluaciones
variables = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34"
]

print("RESUMEN GENERAL DE EVALUACIONES")
print(df[variables].describe().round(2))

print("\nVALORES CERO POR VARIABLE")
for variable in variables:
    cantidad = (df[variable] == 0).sum()
    porcentaje = cantidad / len(df) * 100

    print(
        variable,
        "| Ceros:", cantidad,
        "| Porcentaje:", round(porcentaje, 2), "%"
    )

# 3. Revisar cada módulo y periodo
resumen = df.groupby(
    ["code_module", "code_presentation"]
).agg(
    estudiantes=("id_student", "count"),
    retiradas=("target_future_withdrawal", "sum"),
    programadas_promedio=("evaluaciones_programadas_34", "mean"),
    entregadas_promedio=("evaluaciones_entregadas_34", "mean"),
    sin_programadas=(
        "evaluaciones_programadas_34",
        lambda x: (x == 0).sum()
    ),
    sin_entregas=(
        "evaluaciones_entregadas_34",
        lambda x: (x == 0).sum()
    )
).reset_index()

resumen["porcentaje_sin_programadas"] = (
    resumen["sin_programadas"] / resumen["estudiantes"] * 100
)

resumen["porcentaje_sin_entregas"] = (
    resumen["sin_entregas"] / resumen["estudiantes"] * 100
)

print("\nRESUMEN POR MODULO Y PERIODO")
print(resumen.round(2).to_string(index=False))

# 4. Revisar específicamente GGG
ggg = resumen[resumen["code_module"] == "GGG"]

print("\nDETALLE DEL MODULO GGG")
print(ggg.round(2).to_string(index=False))

# 5. Guardar resultados
resumen.to_csv(
    "outputs/tables/assessment_audit_by_module_v1.csv",
    index=False
)

print("\nArchivo guardado:")
print("outputs/tables/assessment_audit_by_module_v1.csv")
