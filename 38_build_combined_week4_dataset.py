
import pandas as pd

archivo_actividad = "data/interim/week4_enriched_features.parquet"
archivo_evaluaciones = "data/interim/week4_assessment_features_v3.parquet"

actividad = pd.read_parquet(archivo_actividad)
evaluaciones = pd.read_parquet(archivo_evaluaciones)

# Seleccionar las variables de evaluacion
columnas_evaluacion = [
    "code_module",
    "code_presentation",
    "id_student",
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34",
    "promedio_notas_34",
    "evaluaciones_aprobadas_34",
    "proporcion_entregadas_34",
    "entregas_tempranas_totales"
]

evaluaciones = evaluaciones[columnas_evaluacion]

# Combinar por estudiante, modulo y presentacion
datos = pd.merge(
    actividad,
    evaluaciones,
    on=[
        "code_module",
        "code_presentation",
        "id_student"
    ],
    how="left",
    validate="one_to_one"
)

# Guardar sin modificar los archivos anteriores
archivo_salida = "data/interim/week4_combined_features.parquet"
datos.to_parquet(archivo_salida, index=False)

print("CONJUNTO COMBINADO CREADO")
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

print("\nVARIABLES DE EVALUACION")
print(datos[columnas_evaluacion[3:]].describe().round(2).to_string())

print("\nVALORES FALTANTES EN VARIABLES DE EVALUACION")
print(datos[columnas_evaluacion[3:]].isna().sum().to_string())

print("\nARCHIVO GUARDADO EN:")
print(archivo_salida)
