
# 110_resumen_experimentos.py

import pandas as pd
from pathlib import Path

# 1. RUTAS

BASE = Path(__file__).resolve().parent.parent
carpeta = BASE / "outputs" / "tables"

# 2. ARCHIVOS DE RESULTADOS

archivos = {
    "106": "106_evaluacion_final.csv",
    "107": "107_validacion_independiente_calibracion.csv",
    "108": "108_comparacion_baselines.csv",
    "109": "109_evaluacion_capacidad_alertas.csv"
}

# 3. COMPROBAR QUE EXISTAN LOS ARCHIVOS

for nombre, archivo in archivos.items():

    ruta = carpeta / archivo

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo del experimento {nombre}: {ruta}"
        )

# 4. LEER LOS RESULTADOS

df106 = pd.read_csv(carpeta / archivos["106"])
df107 = pd.read_csv(carpeta / archivos["107"])
df108 = pd.read_csv(carpeta / archivos["108"])
df109 = pd.read_csv(carpeta / archivos["109"])

# 5. MOSTRAR RESUMEN DE LOS MODELOS

columnas = [
    "modelo",
    "periodo",
    "estudiantes",
    "AUC",
    "F1",
    "precision",
    "recall",
    "Brier",
    "alertas",
    "porcentaje_alertas"
]

print("\nEXPERIMENTO 106")
print(df106[columnas].round(4).to_string(index=False))

print("\nEXPERIMENTO 107")
print(df107[columnas].round(4).to_string(index=False))

print("\nEXPERIMENTO 108")
print(df108[columnas].round(4).to_string(index=False))

# 6. MOSTRAR RESULTADOS POR CAPACIDAD

print("\nEXPERIMENTO 109: CAPACIDAD DE ATENCION")
print(df109.round(4).to_string(index=False))

# 7. GUARDAR UN RESUMEN UNIFICADO

df106["experimento"] = "106"
df107["experimento"] = "107"
df108["experimento"] = "108"

resumen_modelos = pd.concat(
    [df106, df107, df108],
    ignore_index=True
)

ruta_modelos = carpeta / "110_resumen_modelos.csv"
ruta_capacidad = carpeta / "110_resumen_capacidad.csv"

resumen_modelos.to_csv(ruta_modelos, index=False)
df109.to_csv(ruta_capacidad, index=False)

print("\nARCHIVOS GUARDADOS")
print(ruta_modelos)
print(ruta_capacidad)
