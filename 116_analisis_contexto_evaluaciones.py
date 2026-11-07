
# 116_analisis_contexto_evaluaciones.py

import pandas as pd
from pathlib import Path

# Rutas del proyecto
BASE = Path(__file__).resolve().parents[1]

ARCHIVO_ENTRADA = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
CARPETA_SALIDA = BASE / "outputs" / "tables"

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# Cargar los datos
df = pd.read_parquet(ARCHIVO_ENTRADA)

print("DATOS CARGADOS")
print("Registros:", len(df))
print()

# Verificar las columnas necesarias
columnas = [
    "code_module",
    "code_presentation",
    "evaluaciones_programadas_34",
    "target_future_withdrawal"
]

for columna in columnas:
    if columna not in df.columns:
        raise ValueError(f"No se encontró la columna: {columna}")

# Agrupar por módulo y presentación
resumen = (
    df.groupby(
        ["code_module", "code_presentation"],
        dropna=False
    )
    .agg(
        estudiantes=("target_future_withdrawal", "size"),
        retiros_futuros=("target_future_withdrawal", "sum"),
        valores_evaluaciones=("evaluaciones_programadas_34", "nunique"),
        evaluaciones_min=("evaluaciones_programadas_34", "min"),
        evaluaciones_max=("evaluaciones_programadas_34", "max")
    )
    .reset_index()
)

# Calcular la tasa de abandono
resumen["tasa_abandono"] = (
    resumen["retiros_futuros"] / resumen["estudiantes"]
)

# Comprobar si la variable es constante dentro de cada grupo
resumen["variable_constante"] = (
    resumen["valores_evaluaciones"] <= 1
)

print("RESUMEN POR MODULO Y PRESENTACION")
print(resumen.to_string(index=False))
print()

print("VERIFICACION DE CONSTANCIA")
print(
    "Grupos analizados:",
    len(resumen)
)
print(
    "Grupos con variable constante:",
    resumen["variable_constante"].sum()
)
print(
    "Grupos con más de un valor:",
    (resumen["variable_constante"] == False).sum()
)
print()

# Resumen según la cantidad de evaluaciones programadas
resumen_evaluaciones = (
    df.groupby("evaluaciones_programadas_34", dropna=False)
    .agg(
        grupos=("code_presentation", "count"),
        estudiantes=("target_future_withdrawal", "size"),
        retiros_futuros=("target_future_withdrawal", "sum")
    )
    .reset_index()
)

resumen_evaluaciones["tasa_abandono"] = (
    resumen_evaluaciones["retiros_futuros"]
    / resumen_evaluaciones["estudiantes"]
)

print("RESULTADOS SEGUN EVALUACIONES PROGRAMADAS")
print(resumen_evaluaciones.to_string(index=False))
print()

# Guardar resultados
archivo_grupos = (
    CARPETA_SALIDA / "116_resumen_modulo_presentacion.csv"
)

archivo_evaluaciones = (
    CARPETA_SALIDA / "116_resumen_por_evaluaciones.csv"
)

resumen.to_csv(archivo_grupos, index=False)
resumen_evaluaciones.to_csv(archivo_evaluaciones, index=False)

print("ARCHIVOS GUARDADOS")
print(archivo_grupos)
print(archivo_evaluaciones)
