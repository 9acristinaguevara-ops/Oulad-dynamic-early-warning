
import pandas as pd

# Cargar los archivos originales y las características
assessments = pd.read_csv("data/raw/csv/assessments.csv")
student_assessment = pd.read_csv(
    "data/raw/csv/studentAssessment.csv"
)

df = pd.read_parquet(
    "data/interim/week4_assessment_features_v3.parquet"
)

# Convertir las fechas
assessments["date"] = pd.to_numeric(
    assessments["date"], errors="coerce"
)

student_assessment["date_submitted"] = pd.to_numeric(
    student_assessment["date_submitted"],
    errors="coerce"
)

# Evaluaciones programadas hasta el día 34
programadas = assessments[
    assessments["date"].notna() &
    (assessments["date"] <= 34)
]

print("EVALUACIONES PROGRAMADAS")
print("------------------------------")

print("Evaluaciones programadas hasta día 34:", len(programadas))

print(
    "Módulos y presentaciones con evaluaciones:",
    programadas[["code_module", "code_presentation"]]
    .drop_duplicates()
    .shape[0]
)

print("\nFALTANTES EN LAS CARACTERÍSTICAS")
print("------------------------------")

print(
    "Filas sin evaluaciones programadas:",
    df["evaluaciones_programadas_34"].isna().sum()
)

print(
    "Filas sin promedio de notas:",
    df["promedio_notas_34"].isna().sum()
)

print("\nEJEMPLOS SIN EVALUACIONES PROGRAMADAS")
print("------------------------------")

print(
    df[df["evaluaciones_programadas_34"].isna()]
    [["code_module", "code_presentation",
      "id_student", "evaluaciones_programadas_34",
      "evaluaciones_entregadas_34",
      "promedio_notas_34"]]
    .head(10)
    .to_string(index=False)
)

print("\nEJEMPLOS SIN PROMEDIO DE NOTAS")
print("------------------------------")

print(
    df[df["promedio_notas_34"].isna()]
    [["code_module", "code_presentation",
      "id_student", "evaluaciones_programadas_34",
      "evaluaciones_entregadas_34",
      "promedio_notas_34"]]
    .head(10)
    .to_string(index=False)
)
