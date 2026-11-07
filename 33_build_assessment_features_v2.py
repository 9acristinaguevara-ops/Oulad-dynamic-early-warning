
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/csv")
INTERIM = Path("data/interim")
CORTE = 34

# Leer archivos
evaluaciones = pd.read_csv(
    RAW / "assessments.csv",
    na_values=["?"]
)

entregas = pd.read_csv(
    RAW / "studentAssessment.csv",
    na_values=["?"]
)

datos = pd.read_parquet(
    INTERIM / "corrected_week4_final_dataset.parquet"
)

claves = [
    "code_module",
    "code_presentation",
    "id_student"
]

claves_curso = [
    "code_module",
    "code_presentation"
]

# Evaluaciones programadas hasta el día 34
programadas = evaluaciones[
    evaluaciones["date"].notna()
    & (evaluaciones["date"] <= CORTE)
].copy()

programadas = programadas.groupby(
    claves_curso
).size().reset_index(
    name="evaluaciones_programadas_34"
)

# Relacionar cada entrega con su evaluación
entregas = entregas.merge(
    evaluaciones[
        ["id_assessment", "code_module",
         "code_presentation", "date"]
    ],
    on="id_assessment",
    how="inner",
    validate="many_to_one"
)

# Entregas realizadas hasta el día 34
tempranas = entregas[
    entregas["date_submitted"].notna()
    & (entregas["date_submitted"] <= CORTE)
    & (entregas["is_banked"] == 0)
].copy()

# Resumir actividad académica por estudiante y matrícula
resumen = tempranas.groupby(claves).agg(
    evaluaciones_entregadas_34=("id_assessment", "nunique"),
    promedio_notas_34=("score", "mean"),
    evaluaciones_aprobadas_34=(
        "score", lambda notas: (notas >= 40).sum()
    )
).reset_index()

# Incorporar número de evaluaciones programadas
datos = datos.merge(
    programadas,
    on=claves_curso,
    how="left",
    validate="many_to_one"
)

# Incorporar resultados de evaluaciones entregadas
datos = datos.merge(
    resumen,
    on=claves,
    how="left",
    validate="one_to_one"
)

# Cero entregas no significa nota cero
datos["evaluaciones_entregadas_34"] = (
    datos["evaluaciones_entregadas_34"].fillna(0)
)

datos["evaluaciones_aprobadas_34"] = (
    datos["evaluaciones_aprobadas_34"].fillna(0)
)

datos["proporcion_entregadas_34"] = (
    datos["evaluaciones_entregadas_34"]
    / datos["evaluaciones_programadas_34"].replace(0, pd.NA)
)

# Guardar en un archivo nuevo
salida = INTERIM / "week4_assessment_features_v2.parquet"
datos.to_parquet(salida, index=False)

print("CONJUNTO V2 CREADO")
print("Filas:", len(datos))
print("\nValores faltantes:")
print(datos[[
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34",
    "proporcion_entregadas_34"
]].isna().sum().to_string())

print("\nArchivo guardado:", salida)
