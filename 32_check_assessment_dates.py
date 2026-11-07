
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/csv")

evaluaciones = pd.read_csv(
    RAW / "assessments.csv",
    na_values=["?"]
)

entregas = pd.read_csv(
    RAW / "studentAssessment.csv",
    na_values=["?"]
)

combinado = entregas.merge(
    evaluaciones[
        ["id_assessment", "code_module", "code_presentation", "date"]
    ],
    on="id_assessment",
    how="left",
    validate="many_to_one"
)

print("TOTAL DE ENTREGAS:", len(combinado))

print("\nFECHAS DE ENTREGA FALTANTES:")
print(combinado["date_submitted"].isna().sum())

print("\nEVALUACIONES SIN FECHA PROGRAMADA:")
print(combinado["date"].isna().sum())

print("\nENTREGAS ANTERIORES AL DÍA 0:")
print((combinado["date_submitted"] < 0).sum())

print("\nENTREGAS HASTA EL DÍA 34:")
print(
    (
        combinado["date_submitted"].notna()
        & (combinado["date_submitted"] <= 34)
    ).sum()
)

print("\nENTREGAS DE EVALUACIONES PROGRAMADAS DESPUÉS DEL DÍA 34:")
print(
    (
        combinado["date"].notna()
        & (combinado["date"] > 34)
        & combinado["date_submitted"].notna()
        & (combinado["date_submitted"] <= 34)
    ).sum()
)

print("\nDIFERENCIA ENTRE FECHA DE ENTREGA Y FECHA PROGRAMADA:")
validos = combinado[
    combinado["date"].notna()
    & combinado["date_submitted"].notna()
]

diferencia = validos["date_submitted"] - validos["date"]
print(diferencia.describe().round(2))
