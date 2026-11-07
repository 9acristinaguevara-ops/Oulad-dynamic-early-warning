
import pandas as pd

archivo = "data/interim/weekly_activity.parquet"

actividad = pd.read_parquet(archivo)

print("SEMANAS DISPONIBLES")
print(actividad["week"].value_counts().sort_index())

print("\nRANGO DE SEMANAS")
print("Mínima:", actividad["week"].min())
print("Máxima:", actividad["week"].max())

print("\nREGISTROS DE LAS SEMANAS 0 A 4")
temprana = actividad[actividad["week"].between(0, 4)]

print("Filas:", len(temprana))
print("Estudiantes:", temprana["id_student"].nunique())

print("\nACTIVIDAD POR SEMANA")
print(
    temprana.groupby("week").agg(
        estudiantes=("id_student", "nunique"),
        clics=("weekly_clicks", "sum"),
        registros=("activity_records", "sum")
    )
)

print("\nCOMPROBACIÓN")
print(
    "Hay semanas negativas:",
    (actividad["week"] < 0).any()
)
