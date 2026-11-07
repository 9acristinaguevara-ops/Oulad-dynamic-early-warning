
import pandas as pd

# 1. Cargar los archivos originales
assessments = pd.read_csv("data/raw/csv/assessments.csv")
student_assessment = pd.read_csv(
    "data/raw/csv/studentAssessment.csv"
)

# 2. Convertir las fechas a valores numéricos
assessments["date"] = pd.to_numeric(
    assessments["date"], errors="coerce"
)

student_assessment["date_submitted"] = pd.to_numeric(
    student_assessment["date_submitted"],
    errors="coerce"
)

# 3. Unir las entregas con la información de evaluaciones
datos = pd.merge(
    student_assessment,
    assessments[["id_assessment", "date"]],
    on="id_assessment",
    how="left"
)

# 4. Revisar las entregas hasta el día 34
entregas_hasta_34 = datos[
    datos["date_submitted"] <= 34
]

# 5. Identificar entregas de evaluaciones programadas después del día 34
entregas_tempranas = entregas_hasta_34[
    entregas_hasta_34["date"] > 34
]

# 6. Mostrar resultados
print("REVISIÓN DE FECHAS")
print("------------------------------")

print("Total de entregas:", len(datos))

print(
    "Entregas realizadas hasta el día 34:",
    len(entregas_hasta_34)
)

print(
    "Entregas hasta el día 34 de evaluaciones programadas después del día 34:",
    len(entregas_tempranas)
)

print(
    "Entregas sin fecha programada:",
    datos["date"].isna().sum()
)

print(
    "Entregas sin fecha de envío:",
    datos["date_submitted"].isna().sum()
)

print("\nEJEMPLOS DE ENTREGAS TEMPRANAS")
print("------------------------------")

print(
    entregas_tempranas[
        ["id_assessment", "id_student",
         "date_submitted", "date"]
    ].head(10).to_string(index=False)
)
