
import pandas as pd

ruta_dataset = "data/interim/week4_early_assessment_v1.parquet"
ruta_student = "data/raw/csv/studentAssessment.csv"
ruta_assessments = "data/raw/csv/assessments.csv"

# Cargar archivos
datos = pd.read_parquet(ruta_dataset)
student = pd.read_csv(ruta_student)
assessments = pd.read_csv(ruta_assessments)

student["date_submitted"] = pd.to_numeric(
    student["date_submitted"], errors="coerce"
)

assessments["date"] = pd.to_numeric(
    assessments["date"], errors="coerce"
)

print("COLUMNAS DEL DATASET")
print(datos.columns.tolist())

print("\nCOLUMNAS DE IDENTIFICACIÓN")
print([
    columna for columna in
    ["id_student", "code_module", "code_presentation"]
    if columna in datos.columns
])

print("\nVARIABLES DE PARTICIPACIÓN")
print([
    columna for columna in datos.columns
    if "evaluaciones_" in columna
])

print("\nPRIMERAS FILAS DE LAS VARIABLES")
columnas = [
    columna for columna in [
        "evaluaciones_programadas_34",
        "evaluaciones_entregadas_34"
    ]
    if columna in datos.columns
]

print(datos[columnas].head(10))

# Revisar si hay valores ausentes
print("\nVALORES AUSENTES")
print(datos[columnas].isna().sum())

# Resumen por módulo y presentación
if all(
    columna in datos.columns for columna in [
        "code_module",
        "code_presentation"
    ]
):
    print("\nRESUMEN POR MÓDULO Y PRESENTACIÓN")
    print(
        datos.groupby(
            ["code_module", "code_presentation"]
        )[columnas].mean().round(2)
    )
