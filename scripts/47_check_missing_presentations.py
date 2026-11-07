
import pandas as pd

# Cargar los archivos
df = pd.read_parquet(
    "data/interim/week4_assessment_features_v3.parquet"
)

assessments = pd.read_csv("data/raw/csv/assessments.csv")

assessments["date"] = pd.to_numeric(
    assessments["date"], errors="coerce"
)

# Obtener todas las combinaciones existentes en el dataset
presentaciones = df[
    ["code_module", "code_presentation"]
].drop_duplicates()

# Obtener las combinaciones con evaluaciones hasta el día 34
programadas = assessments[
    assessments["date"].notna() &
    (assessments["date"] <= 34)
]

combinaciones_con_evaluaciones = programadas[
    ["code_module", "code_presentation"]
].drop_duplicates()

# Identificar las combinaciones sin evaluaciones hasta el día 34
comparacion = presentaciones.merge(
    combinaciones_con_evaluaciones,
    on=["code_module", "code_presentation"],
    how="left",
    indicator=True
)

sin_evaluaciones = comparacion[
    comparacion["_merge"] == "left_only"
]

print("PRESENTACIONES SIN EVALUACIONES HASTA EL DÍA 34")
print("------------------------------------------------")

print(sin_evaluaciones.to_string(index=False))

print("\nMATRÍCULAS EN ESAS PRESENTACIONES")
print("------------------------------------------------")

for _, fila in sin_evaluaciones.iterrows():

    modulo = fila["code_module"]
    presentacion = fila["code_presentation"]

    grupo = df[
        (df["code_module"] == modulo) &
        (df["code_presentation"] == presentacion)
    ]

    print("\nMódulo:", modulo)
    print("Presentación:", presentacion)
    print("Número de matrículas:", len(grupo))
    print(
        "Sin evaluaciones programadas:",
        grupo["evaluaciones_programadas_34"].isna().sum()
    )
    print(
        "Sin promedio de notas:",
        grupo["promedio_notas_34"].isna().sum()
    )
