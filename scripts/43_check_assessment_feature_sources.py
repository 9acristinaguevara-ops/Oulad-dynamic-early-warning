
import pandas as pd

# Cargar los archivos
assessments = pd.read_csv("data/raw/csv/assessments.csv")
student_assessment = pd.read_csv(
    "data/raw/csv/studentAssessment.csv"
)

features = pd.read_parquet(
    "data/interim/week4_assessment_features_v3.parquet"
)

# Convertir fechas y notas a valores numéricos
assessments["date"] = pd.to_numeric(
    assessments["date"], errors="coerce"
)

student_assessment["date_submitted"] = pd.to_numeric(
    student_assessment["date_submitted"],
    errors="coerce"
)

student_assessment["score"] = pd.to_numeric(
    student_assessment["score"], errors="coerce"
)

# Unir las entregas con las fechas programadas
datos = pd.merge(
    student_assessment,
    assessments[["id_assessment", "date"]],
    on="id_assessment",
    how="left"
)

# Entregas realizadas hasta el día 34
hasta_34 = datos[
    datos["date_submitted"] <= 34
]

# Entregas de evaluaciones programadas hasta el día 34
programadas_hasta_34 = hasta_34[
    (hasta_34["date"].isna()) |
    (hasta_34["date"] <= 34)
]

# Mostrar resumen
print("REVISIÓN DE NOTAS Y ENTREGAS")
print("------------------------------")

print("Entregas hasta el día 34:", len(hasta_34))

print(
    "Entregas de evaluaciones programadas hasta el día 34:",
    len(programadas_hasta_34)
)

print(
    "Notas disponibles en esos registros:",
    programadas_hasta_34["score"].notna().sum()
)

print(
    "Notas faltantes en esos registros:",
    programadas_hasta_34["score"].isna().sum()
)

print("\nVARIABLES DEL DATASET V3")
print("------------------------------")

columnas = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34",
    "proporcion_entregadas_34",
    "entregas_tempranas_totales"
]

for columna in columnas:
    if columna in features.columns:
        print(
            columna,
            "| faltantes:",
            features[columna].isna().sum(),
            "| valores únicos:",
            features[columna].nunique()
        )
