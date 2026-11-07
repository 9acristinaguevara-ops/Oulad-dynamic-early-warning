
# 134_validacion_persistencia_agrupada.py

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

# =====================================================
# 3. CONVERTIR EL PANEL A FORMATO LONGITUDINAL
# =====================================================

registros = []

for semana in range(5):

    columna = f"clics_semana_{semana}"

    temporal = df[
        [
            "code_module",
            "code_presentation",
            "id_student",
            columna
        ]
    ].copy()

    temporal = temporal.rename(
        columns={columna: "clics"}
    )

    temporal["semana"] = semana

    registros.append(temporal)

longitudinal = pd.concat(
    registros,
    ignore_index=True
)

# Excluir semanas sin registro.
# No se convierten en cero.

longitudinal = longitudinal.dropna(
    subset=["clics"]
).copy()

longitudinal["log_clics"] = np.log1p(
    longitudinal["clics"]
)

longitudinal["grupo"] = (
    longitudinal["code_module"].astype(str)
    + "_"
    + longitudinal["code_presentation"].astype(str)
)

# =====================================================
# 4. NORMALIZACION CONSISTENTE POR GRUPO
# =====================================================

media_grupo = (
    longitudinal.groupby("grupo")["log_clics"]
    .transform("mean")
)

desv_grupo = (
    longitudinal.groupby("grupo")["log_clics"]
    .transform("std")
)

longitudinal["actividad_relativa"] = (
    longitudinal["log_clics"] - media_grupo
) / desv_grupo

longitudinal = longitudinal.replace(
    [np.inf, -np.inf], np.nan
)

longitudinal = longitudinal.dropna(
    subset=["actividad_relativa"]
)

# =====================================================
# 5. CONSTRUIR PARES DE SEMANAS CONSECUTIVAS
# =====================================================

claves = [
    "code_module",
    "code_presentation",
    "id_student"
]

transiciones = []

for semana in range(4):

    actual = longitudinal[
        longitudinal["semana"] == semana
    ][claves + ["actividad_relativa"]].copy()

    siguiente = longitudinal[
        longitudinal["semana"] == semana + 1
    ][claves + ["actividad_relativa"]].copy()

    actual = actual.rename(
        columns={
            "actividad_relativa": "actividad_actual"
        }
    )

    siguiente = siguiente.rename(
        columns={
            "actividad_relativa": "actividad_siguiente"
        }
    )

    par = actual.merge(
        siguiente,
        on=claves,
        how="inner"
    )

    par["semana"] = semana

    transiciones.append(par)

pares = pd.concat(
    transiciones,
    ignore_index=True
)

pares["grupo"] = (
    pares["code_module"].astype(str)
    + "_"
    + pares["code_presentation"].astype(str)
)

print("\nTRANSICIONES DISPONIBLES")
print("Total de pares:", len(pares))
print("Estudiantes:", pares["id_student"].nunique())
print("Grupos:", pares["grupo"].nunique())

print("\nPARES POR SEMANA")
print(pares["semana"].value_counts().sort_index())

# =====================================================
# 6. CORRELACION DESCRIPTIVA
# =====================================================

correlacion = pares[
    ["actividad_actual", "actividad_siguiente"]
].corr().iloc[0, 1]

print("\nCORRELACION")
print(round(correlacion, 4))

# =====================================================
# 7. MODELO CON ERRORES AGRUPADOS POR ESTUDIANTE
# =====================================================

modelo = smf.ols(
    """
    actividad_siguiente ~ actividad_actual
    + C(grupo)
    + C(semana)
    """,
    data=pares
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": pares["id_student"]
    }
)

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
        "error_estandar_agrupado",
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
    "134_validacion_persistencia_agrupada.csv"
)

resumen.to_csv(
    archivo_salida,
    index=False
)

print("\nRESULTADOS GUARDADOS EN:")
print(archivo_salida)

print("\nEXPERIMENTO 134 FINALIZADO")
