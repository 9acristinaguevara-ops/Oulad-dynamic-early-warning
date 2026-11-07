
import pandas as pd

# Archivos de entrada
archivo_base = "data/interim/corrected_week4_final_dataset.parquet"
archivo_entregas = "data/raw/csv/studentAssessment.csv"
archivo_evaluaciones = "data/raw/csv/assessments.csv"

# Cargar los datos
base = pd.read_parquet(archivo_base)
entregas = pd.read_csv(archivo_entregas)
evaluaciones = pd.read_csv(archivo_evaluaciones)


# Convertir las fechas a numeros
evaluaciones["date"] = pd.to_numeric(
    evaluaciones["date"], errors="coerce"
)

entregas["date_submitted"] = pd.to_numeric(
    entregas["date_submitted"], errors="coerce"
)

# Convertir las notas a valores numericos
entregas["score"] = pd.to_numeric(
    entregas["score"], errors="coerce"
)

# Excluir entregas bancadas
entregas = entregas[entregas["is_banked"] == 0]


# Excluir entregas bancadas
entregas = entregas[entregas["is_banked"] == 0]

# Unir entregas con informacion de las evaluaciones
datos = pd.merge(
    entregas,
    evaluaciones,
    on="id_assessment",
    how="left"
)

# Evaluaciones programadas hasta el dia 34
programadas = evaluaciones[
    evaluaciones["date"].notna()
    & (evaluaciones["date"] <= 34)
]

conteo_programadas = programadas.groupby(
    ["code_module", "code_presentation"]
)["id_assessment"].nunique().reset_index()

conteo_programadas = conteo_programadas.rename(
    columns={"id_assessment": "evaluaciones_programadas_34"}
)

# Entregas realizadas hasta el dia 34
tempranas = datos[
    datos["date_submitted"].notna()
    & (datos["date_submitted"] <= 34)
].copy()

# Separar entregas de evaluaciones programadas hasta el dia 34
entregas_programadas = tempranas[
    tempranas["date"].notna()
    & (tempranas["date"] <= 34)
].copy()

# Calcular variables por estudiante y asignatura
entregadas = entregas_programadas.groupby(
    ["code_module", "code_presentation", "id_student"]
).agg(
    evaluaciones_entregadas_34=("id_assessment", "nunique"),
    promedio_notas_34=("score", "mean"),
    evaluaciones_aprobadas_34=("score", lambda x: (x >= 40).sum())
).reset_index()

# Contar todas las entregas tempranas, incluso de evaluaciones futuras
todas_entregas = tempranas.groupby(
    ["code_module", "code_presentation", "id_student"]
)["id_assessment"].nunique().reset_index()

todas_entregas = todas_entregas.rename(
    columns={"id_assessment": "entregas_tempranas_totales"}
)

# Unir las variables con la base principal
resultado = pd.merge(
    base,
    conteo_programadas,
    on=["code_module", "code_presentation"],
    how="left"
)

resultado = pd.merge(
    resultado,
    entregadas,
    on=["code_module", "code_presentation", "id_student"],
    how="left"
)

resultado = pd.merge(
    resultado,
    todas_entregas,
    on=["code_module", "code_presentation", "id_student"],
    how="left"
)

# Reemplazar por cero solo los conteos sin entregas
columnas_conteo = [
    "evaluaciones_entregadas_34",
    "evaluaciones_aprobadas_34",
    "entregas_tempranas_totales"
]

for columna in columnas_conteo:
    resultado[columna] = resultado[columna].fillna(0)

# Calcular la proporcion de cumplimiento
resultado["proporcion_entregadas_34"] = (
    resultado["evaluaciones_entregadas_34"]
    / resultado["evaluaciones_programadas_34"]
)

# Guardar el nuevo conjunto
resultado.to_parquet(
    "data/interim/week4_assessment_features_v3.parquet",
    index=False
)

print("ARCHIVO V3 CREADO")
print("Filas:", len(resultado))
print("Columnas:", len(resultado.columns))
print("\nValores faltantes:")
print(
    resultado[
        [
            "evaluaciones_programadas_34",
            "evaluaciones_entregadas_34",
            "promedio_notas_34",
            "evaluaciones_aprobadas_34",
            "proporcion_entregadas_34",
            "entregas_tempranas_totales"
        ]
    ].isna().sum()
)

print("\nProporcion de cumplimiento:")
print(
    resultado["proporcion_entregadas_34"]
    .describe()
    .round(3)
)

print("\nProporciones mayores que 1:")
print(
    (
        resultado["proporcion_entregadas_34"] > 1
    ).sum()
)

print("\nEtiquetas:")
print(resultado["target_future_withdrawal"].value_counts())
