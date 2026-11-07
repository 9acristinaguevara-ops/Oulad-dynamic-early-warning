
import pandas as pd

ruta_student = "data/raw/csv/studentAssessment.csv"
ruta_assessments = "data/raw/csv/assessments.csv"

# Leer archivos
student = pd.read_csv(ruta_student)
assessments = pd.read_csv(ruta_assessments)

# Convertir las fechas a números
student["date_submitted"] = pd.to_numeric(
    student["date_submitted"], errors="coerce"
)

assessments["date"] = pd.to_numeric(
    assessments["date"], errors="coerce"
)

print("ARCHIVOS CARGADOS")
print("Registros de entregas:", len(student))
print("Evaluaciones:", len(assessments))

# Unir los archivos
datos = pd.merge(
    student,
    assessments,
    on="id_assessment",
    how="left"
)

print("\nCOLUMNAS DISPONIBLES")
print(datos.columns.tolist())

# Entregas realizadas hasta el día 34
entregas_34 = datos[
    datos["date_submitted"] <= 34
]

print("\nAUDITORÍA DE ENTREGAS HASTA EL DÍA 34")
print("Entregas hasta el día 34:", len(entregas_34))

# Evaluaciones con fecha límite posterior al día 34
fecha_limite_futura = entregas_34[
    entregas_34["date"] > 34
]

# Evaluaciones sin fecha límite registrada
fecha_limite_ausente = entregas_34[
    entregas_34["date"].isna()
]

print(
    "Con fecha límite posterior al día 34:",
    len(fecha_limite_futura)
)

print(
    "Sin fecha límite registrada:",
    len(fecha_limite_ausente)
)

# Revisar calificaciones
sin_calificacion = entregas_34[
    entregas_34["score"].isna()
]

con_calificacion = entregas_34[
    entregas_34["score"].notna()
]

print("\nAUDITORÍA DE CALIFICACIONES")
print("Con calificación registrada:", len(con_calificacion))
print("Sin calificación registrada:", len(sin_calificacion))

# Fechas de entrega negativas
fechas_negativas = datos[
    datos["date_submitted"] < 0
]

print("\nFECHAS NEGATIVAS")
print("Entregas anteriores al día 0:", len(fechas_negativas))

# Fechas de entrega ausentes
fechas_ausentes = datos[
    datos["date_submitted"].isna()
]

print("Entregas sin fecha registrada:", len(fechas_ausentes))

# Resumen
print("\nRESUMEN")
print("Total de entregas:", len(datos))
print("Entregas hasta el día 34:", len(entregas_34))

if len(entregas_34) > 0:
    porcentaje = len(sin_calificacion) / len(entregas_34) * 100
    print(
        "Porcentaje sin calificación registrada:",
        round(porcentaje, 2),
        "%"
    )
