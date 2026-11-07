
import pandas as pd

archivo_entregas = "data/raw/csv/studentAssessment.csv"
archivo_evaluaciones = "data/raw/csv/assessments.csv"

entregas = pd.read_csv(archivo_entregas)
evaluaciones = pd.read_csv(archivo_evaluaciones)

# Convertir las fechas a valores numericos
evaluaciones["date"] = pd.to_numeric(
    evaluaciones["date"],
    errors="coerce"
)

entregas["date_submitted"] = pd.to_numeric(
    entregas["date_submitted"],
    errors="coerce"
)

datos = pd.merge(
    entregas,
    evaluaciones,
    on="id_assessment",
    how="left"
)

# Revisar entregas realizadas hasta el dia 34
tempranas = datos[datos["date_submitted"] <= 34]

print("ENTREGAS HASTA EL DIA 34")
print("Total:", len(tempranas))

print("\nFECHA PROGRAMADA DE ESAS EVALUACIONES")
print(tempranas["date"].describe())

print("\nENTREGAS DE EVALUACIONES PROGRAMADAS DESPUES DEL DIA 34")
print(
    (
        tempranas["date"].notna()
        & (tempranas["date"] > 34)
    ).sum()
)

print("\nENTREGAS SIN FECHA PROGRAMADA")
print(tempranas["date"].isna().sum())

print("\nTIPOS DE EVALUACION")
print(tempranas["assessment_type"].value_counts())
