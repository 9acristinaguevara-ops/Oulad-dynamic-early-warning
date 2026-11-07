
import pandas as pd
from pathlib import Path

BASE = Path(".")
RAW = BASE / "data" / "raw" / "csv"
INTERIM = BASE / "data" / "interim"

CORTE = 34

# Leer las evaluaciones programadas
evaluaciones = pd.read_csv(
    RAW / "assessments.csv",
    na_values=["?"]
)

# Leer las entregas de los estudiantes
entregas = pd.read_csv(
    RAW / "studentAssessment.csv",
    na_values=["?"]
)

# Leer el conjunto corregido
datos = pd.read_parquet(
    INTERIM / "corrected_week4_final_dataset.parquet"
)

# Conservar solo evaluaciones programadas hasta el día 34
evaluaciones_tempranas = evaluaciones[
    evaluaciones["date"].notna()
    & (evaluaciones["date"] <= CORTE)
].copy()

# Contar evaluaciones programadas por módulo y presentación
claves_curso = ["code_module", "code_presentation"]

programadas = evaluaciones_tempranas.groupby(
    claves_curso
).size().reset_index(name="evaluaciones_programadas_34")

# Relacionar las entregas con las evaluaciones
entregas = entregas.merge(
    evaluaciones[
        ["id_assessment", "code_module", "code_presentation", "date"]
    ],
    on="id_assessment",
    how="inner",
    validate="many_to_one"
)

# Usar solo evaluaciones programadas hasta el corte
# y entregas realizadas hasta el día 34
entregas_tempranas = entregas[
    entregas["date"].notna()
    & (entregas["date"] <= CORTE)
    & entregas["date_submitted"].notna()
    & (entregas["date_submitted"] <= CORTE)
    & (entregas["is_banked"] == 0)
].copy()

# Calcular características por estudiante y matrícula
claves = [
    "code_module",
    "code_presentation",
    "id_student"
]

resumen = entregas_tempranas.groupby(claves).agg(
    evaluaciones_entregadas_34=("id_assessment", "nunique"),
    promedio_notas_34=("score", "mean"),
    evaluaciones_aprobadas_34=("score", lambda notas: (notas >= 40).sum())
).reset_index()

# Añadir el número de evaluaciones programadas
datos = datos.merge(
    programadas,
    on=claves_curso,
    how="left",
    validate="many_to_one"
)

# Añadir las características de cada estudiante
datos = datos.merge(
    resumen,
    on=claves,
    how="left",
    validate="one_to_one"
)

# Los estudiantes sin entregas conocidas tienen cero entregas,
# pero no se supone que tengan una calificación de cero
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

SALIDA = INTERIM / "week4_assessment_features.parquet"
datos.to_parquet(SALIDA, index=False)

print("VARIABLES DE EVALUACIONES CREADAS")
print("Filas:", len(datos))
print("\nNuevas variables:")
print([
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34",
    "proporcion_entregadas_34"
])

nuevas = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34",
    "proporcion_entregadas_34"
]

print("\nValores faltantes:")
print(datos[nuevas].isna().sum().to_string())

print("\nArchivo guardado:", SALIDA)
