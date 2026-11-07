
import pandas as pd
from pathlib import Path

CARPETA = Path("data/raw/csv")

archivos = [
    "assessments.csv",
    "studentAssessment.csv",
    "courses.csv"
]

for nombre in archivos:
    ruta = CARPETA / nombre

    datos = pd.read_csv(
        ruta,
        na_values=["?"],
        nrows=5
    )

    print("\n" + "=" * 60)
    print("ARCHIVO:", nombre)
    print("COLUMNAS:")
    print(list(datos.columns))

    print("\nPRIMEROS REGISTROS:")
    print(datos.to_string(index=False))

    print("\nTIPOS DE DATOS:")
    print(datos.dtypes.to_string())
