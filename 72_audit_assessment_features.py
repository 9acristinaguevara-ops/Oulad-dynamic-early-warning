
import pandas as pd

# 1. Cargar el conjunto de datos
ruta = "data/interim/week4_combined_features_v4.parquet"
df = pd.read_parquet(ruta)

print("AUDITORIA DE VARIABLES DE EVALUACIONES")
print("Filas:", len(df))
print("Columnas:", len(df.columns))


# 2. Revisar las variables de evaluaciones
variables = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34"
]

for variable in variables:
    if variable in df.columns:
        print("\nVariable:", variable)
        print("Valores distintos:", df[variable].nunique())
        print("Valores faltantes:", df[variable].isna().sum())
        print("Valor mínimo:", df[variable].min())
        print("Valor máximo:", df[variable].max())
        print("Promedio:", df[variable].mean())
    else:
        print("\nNo se encontró la variable:", variable)


# 3. Resumir las variables por módulo
print("\nRESUMEN POR MODULO")

columnas = ["code_module"] + [
    variable for variable in variables if variable in df.columns
]

resumen = df[columnas].groupby("code_module").agg(
    ["mean", "min", "max", "nunique"]
)

print(resumen)


# 4. Revisar específicamente GGG
print("\nREVISION DEL MODULO GGG")

ggg = df[df["code_module"] == "GGG"]

for variable in variables:
    if variable in ggg.columns:
        print("\nVariable:", variable)
        print(ggg[variable].value_counts(dropna=False).head(10))


# 5. Guardar resumen
salida = "outputs/tables/assessment_features_audit_v1.csv"

resumen.to_csv(salida)

print("\nArchivo guardado en:", salida)
print("AUDITORIA FINALIZADA")
