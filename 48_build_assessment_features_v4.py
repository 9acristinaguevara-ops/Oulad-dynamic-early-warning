
import pandas as pd

# Cargar las características anteriores
df = pd.read_parquet(
    "data/interim/week4_assessment_features_v3.parquet"
)

assessments = pd.read_csv("data/raw/csv/assessments.csv")

assessments["date"] = pd.to_numeric(
    assessments["date"], errors="coerce"
)

# Contar evaluaciones programadas hasta el día 34
programadas = assessments[
    assessments["date"].notna() &
    (assessments["date"] <= 34)
]

conteo = programadas.groupby(
    ["code_module", "code_presentation"]
)["id_assessment"].count().reset_index()

conteo = conteo.rename(
    columns={"id_assessment": "evaluaciones_programadas_34"}
)

# Unir el conteo con las características existentes
df = df.drop(columns=["evaluaciones_programadas_34"])

df = pd.merge(
    df,
    conteo,
    on=["code_module", "code_presentation"],
    how="left"
)

# Las presentaciones sin evaluaciones programadas reciben cero
df["evaluaciones_programadas_34"] = (
    df["evaluaciones_programadas_34"].fillna(0)
)

# Recalcular la proporción de entregas
df["proporcion_entregadas_34"] = 0.0

hay_evaluaciones = df["evaluaciones_programadas_34"] > 0

df.loc[
    hay_evaluaciones,
    "proporcion_entregadas_34"
] = (
    df.loc[hay_evaluaciones, "evaluaciones_entregadas_34"] /
    df.loc[hay_evaluaciones, "evaluaciones_programadas_34"]
)

# Guardar una nueva versión
df.to_parquet(
    "data/interim/week4_assessment_features_v4.parquet",
    index=False
)

print("Dataset V4 creado correctamente")
print("Filas:", len(df))
print(
    "Faltantes en evaluaciones programadas:",
    df["evaluaciones_programadas_34"].isna().sum()
)
print(
    "Faltantes en proporción de entregas:",
    df["proporcion_entregadas_34"].isna().sum()
)
print(
    "Matrículas sin evaluaciones programadas:",
    (df["evaluaciones_programadas_34"] == 0).sum()
)
print(
    "Faltantes en promedio de notas:",
    df["promedio_notas_34"].isna().sum()
)
