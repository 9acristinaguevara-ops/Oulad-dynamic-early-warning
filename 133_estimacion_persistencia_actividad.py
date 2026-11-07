
# 133_estimacion_persistencia_actividad.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# =====================================================
# 1. RUTAS
# =====================================================

BASE = Path(__file__).resolve().parent.parent

ARCHIVO_ENTRADA = (
    BASE / "data" / "interim" /
    "weekly_panel_weeks0_4.parquet"
)

CARPETA_SALIDA = BASE / "outputs" / "tables"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. CARGAR DATOS
# =====================================================

df = pd.read_parquet(ARCHIVO_ENTRADA)

print("DIMENSIONES DEL PANEL")
print(df.shape)

print("\nCOLUMNAS")
print(df.columns.tolist())

# =====================================================
# 3. CONSTRUIR LAS TRANSICIONES SEMANALES
# =====================================================

transiciones = []

for semana in range(4):

    columna_actual = f"clics_semana_{semana}"
    columna_siguiente = f"clics_semana_{semana + 1}"

    temporal = df[
        [
            "code_module",
            "code_presentation",
            "id_student",
            columna_actual,
            columna_siguiente
        ]
    ].copy()

    temporal = temporal.dropna(
        subset=[columna_actual, columna_siguiente]
    )

    temporal = temporal.rename(
        columns={
            columna_actual: "clics_actuales",
            columna_siguiente: "clics_siguientes"
        }
    )

    temporal["semana"] = semana

    transiciones.append(temporal)

pares = pd.concat(transiciones, ignore_index=True)

pares["grupo"] = (
    pares["code_module"].astype(str)
    + "_"
    + pares["code_presentation"].astype(str)
)

print("\nTRANSICIONES DISPONIBLES")
print("Total de pares:", len(pares))
print("Estudiantes:", pares["id_student"].nunique())
print("Grupos:", pares["grupo"].nunique())

print("\nPARES POR TRANSICION")
print(pares["semana"].value_counts().sort_index())

# =====================================================
# 4. TRANSFORMACION LOGARITMICA
# =====================================================

pares["log_actual"] = np.log1p(pares["clics_actuales"])
pares["log_siguiente"] = np.log1p(pares["clics_siguientes"])

# =====================================================
# 5. NORMALIZACION POR GRUPO
# =====================================================

media_grupo = pares.groupby("grupo")["log_actual"].transform("mean")
desv_grupo = pares.groupby("grupo")["log_actual"].transform("std")

pares["actividad_actual"] = (
    pares["log_actual"] - media_grupo
) / desv_grupo

media_siguiente = pares.groupby("grupo")["log_siguiente"].transform("mean")
desv_siguiente = pares.groupby("grupo")["log_siguiente"].transform("std")

pares["actividad_siguiente"] = (
    pares["log_siguiente"] - media_siguiente
) / desv_siguiente

pares = pares.replace([np.inf, -np.inf], np.nan)

pares = pares.dropna(
    subset=["actividad_actual", "actividad_siguiente"]
)

# =====================================================
# 6. CORRELACION ENTRE SEMANAS
# =====================================================

correlacion = pares[
    ["actividad_actual", "actividad_siguiente"]
].corr().iloc[0, 1]

print("\nCORRELACION ENTRE ACTIVIDAD CONSECUTIVA")
print("Correlacion:", round(correlacion, 4))

# =====================================================
# 7. MODELO DE PERSISTENCIA
# =====================================================

modelo = smf.ols(
    "actividad_siguiente ~ actividad_actual + C(grupo)",
    data=pares
).fit(cov_type="HC3")

print("\nRESULTADOS DEL MODELO")
print(modelo.summary())

# =====================================================
# 8. GUARDAR RESULTADOS
# =====================================================

resumen = pd.DataFrame({
    "indicador": [
        "observaciones",
        "estudiantes",
        "grupos",
        "correlacion",
        "persistencia_rho",
        "error_estandar",
        "p_valor",
        "r_cuadrado"
    ],
    "valor": [
        len(pares),
        pares["id_student"].nunique(),
        pares["grupo"].nunique(),
        correlacion,
        modelo.params["actividad_actual"],
        modelo.bse["actividad_actual"],
        modelo.pvalues["actividad_actual"],
        modelo.rsquared
    ]
})

archivo_salida = (
    CARPETA_SALIDA /
    "133_estimacion_persistencia_actividad.csv"
)

resumen.to_csv(archivo_salida, index=False)

print("\nRESUMEN GUARDADO EN:")
print(archivo_salida)

print("\nEXPERIMENTO 133 FINALIZADO")
