
import pandas as pd

archivo = "data/interim/week4_assessment_features_v3.parquet"
datos = pd.read_parquet(archivo)

print("TAMANO DEL CONJUNTO")
print("Filas:", len(datos))
print("Columnas:", len(datos.columns))

print("\nDUPLICADOS POR MATRICULA")
print(
    datos.duplicated(
        subset=[
            "code_module",
            "code_presentation",
            "id_student"
        ]
    ).sum()
)

print("\nESTADISTICAS DE VARIABLES")
columnas = [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34",
    "proporcion_entregadas_34",
    "entregas_tempranas_totales"
]

print(datos[columnas].describe().round(2).to_string())

print("\nEVALUACIONES APROBADAS MAYORES QUE ENTREGADAS")
print(
    (
        datos["evaluaciones_aprobadas_34"]
        > datos["evaluaciones_entregadas_34"]
    ).sum()
)

print("\nPROPORCIONES MAYORES QUE 1")
print(
    (datos["proporcion_entregadas_34"] > 1).sum()
)

print("\nVALORES FALTANTES")
print(datos[columnas].isna().sum().to_string())

print("\nDISTRIBUCION DEL TARGET")
print(datos["target_future_withdrawal"].value_counts())
