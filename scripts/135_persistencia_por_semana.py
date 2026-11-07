
# 135_persistencia_por_semana.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm
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
# 3. CONVERTIR A FORMATO LONGITUDINAL
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

# No convertir semanas faltantes en cero.

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
# 4. NORMALIZAR CON UNA ESCALA COMUN POR GRUPO
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
# 5. CONSTRUIR TRANSICIONES CONSECUTIVAS
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

print("\nTRANSICIONES")
print("Observaciones:", len(pares))
print("Estudiantes:", pares["id_student"].nunique())
print("Grupos:", pares["grupo"].nunique())

print("\nOBSERVACIONES POR SEMANA")
print(pares["semana"].value_counts().sort_index())

# =====================================================
# 6. MODELO CON PERSISTENCIA VARIABLE
# =====================================================

modelo = smf.ols(
    """
    actividad_siguiente ~
    actividad_actual * C(semana)
    + C(grupo)
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
# 7. CALCULAR PERSISTENCIA POR TRANSICION
# =====================================================

parametros = modelo.params
covarianza = modelo.cov_params()

resultados = []

for semana in range(4):

    # Construir combinación lineal de coeficientes.

    contraste = pd.Series(
        0.0,
        index=parametros.index
    )

    contraste["actividad_actual"] = 1.0

    if semana > 0:

        nombre = (
            f"actividad_actual:C(semana)[T.{semana}]"
        )

        contraste[nombre] = 1.0

    # Coeficiente de persistencia de esta semana.

    rho = float(contraste @ parametros)

    # Error estándar de la combinación lineal.

    varianza = float(
        contraste @ covarianza @ contraste
    )

    error = np.sqrt(max(varianza, 0))

    z = rho / error

    p_valor = 2 * norm.sf(abs(z))

    limite_inferior = rho - 1.96 * error
    limite_superior = rho + 1.96 * error

    resultados.append({
        "transicion": f"{semana} a {semana + 1}",
        "persistencia_rho": rho,
        "error_estandar": error,
        "z": z,
        "p_valor": p_valor,
        "ic95_inferior": limite_inferior,
        "ic95_superior": limite_superior
    })

tabla = pd.DataFrame(resultados)

print("\nPERSISTENCIA POR TRANSICION")
print(tabla.to_string(index=False))

# =====================================================
# 8. GUARDAR RESULTADOS
# =====================================================

archivo_salida = (
    CARPETA_SALIDA /
    "135_persistencia_por_semana.csv"
)

tabla.to_csv(
    archivo_salida,
    index=False
)

print("\nRESULTADOS GUARDADOS EN:")
print(archivo_salida)

print("\nEXPERIMENTO 135 FINALIZADO")
