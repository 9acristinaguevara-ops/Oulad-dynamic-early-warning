
# 140_persistencia_por_abandono.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from pathlib import Path
from scipy.stats import norm, chi2

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


# =====================================================
# 2. CARGAR DATOS Y RECUPERAR TARGET
# =====================================================


# =====================================================
# 2. CARGAR DATOS Y RECUPERAR TARGET
# =====================================================

df = pd.read_parquet(ARCHIVO_ENTRADA)

ARCHIVO_TARGET = (
    BASE / "data" / "interim" /
    "week4_combined_features_v4.parquet"
)

df_target = pd.read_parquet(ARCHIVO_TARGET)

claves = [
    "code_module",
    "code_presentation",
    "id_student"
]

print("DIMENSIONES DEL PANEL SEMANAL")
print(df.shape)

print("\nDIMENSIONES DEL ARCHIVO DE TARGET")
print(df_target.shape)

columnas_necesarias = claves + [
    "target_future_withdrawal"
]

faltantes = [
    columna for columna in columnas_necesarias
    if columna not in df_target.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas en el archivo de target: {faltantes}"
    )

if df_target.duplicated(subset=claves).any():
    raise ValueError(
        "El archivo de target contiene claves duplicadas."
    )

df_target = df_target[columnas_necesarias].copy()

df_target = df_target.rename(
    columns={
        "target_future_withdrawal": "target"
    }
)

df = df.merge(
    df_target,
    on=claves,
    how="inner",
    validate="one_to_one"
)

print("\nPANEL FINAL CON TARGET")
print(df.shape)

print("\nDISTRIBUCION DEL TARGET")
print(df["target"].value_counts(dropna=False))

if df["target"].isna().any():
    raise ValueError(
        "Existen valores ausentes en target."
    )

if len(df) != 27135:
    raise ValueError(
        "El número de registros después del merge no coincide "
        "con los 27135 registros esperados."
    )


# =====================================================
# 3. CONSTRUIR FORMATO LONGITUDINAL
# =====================================================

registros = []

for semana in range(5):

    columna = f"clics_semana_{semana}"

    temporal = df[
        claves + ["target", columna]
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

# No convertir semanas sin registros en cero.
longitudinal = longitudinal.dropna(
    subset=["clics", "target"]
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
# 4. NORMALIZAR ACTIVIDAD POR GRUPO
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
# 5. CONSTRUIR PARES CONSECUTIVOS
# =====================================================

transiciones = []

for semana in range(4):

    actual = longitudinal[
        longitudinal["semana"] == semana
    ][
        claves + ["target", "actividad_relativa"]
    ].copy()

    siguiente = longitudinal[
        longitudinal["semana"] == semana + 1
    ][
        claves + ["actividad_relativa"]
    ].copy()

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
        how="inner",
        validate="one_to_one"
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

pares["target"] = pares["target"].astype(int)

print("\nDATOS DE TRANSICIONES")
print("Observaciones:", len(pares))
print("Estudiantes:", pares["id_student"].nunique())
print(
    "Estudiantes-curso:",
    pares[claves].drop_duplicates().shape[0]
)

print("\nDISTRIBUCION DEL TARGET EN LAS TRANSICIONES")
print(pares["target"].value_counts())

# =====================================================
# 6. CREAR VARIABLES DE INTERACCION
# =====================================================

# Semana 0 es la categoría de referencia.

for semana in range(1, 4):

    pares[f"w{semana}"] = (
        pares["semana"] == semana
    ).astype(int)

    pares[f"target_w{semana}"] = (
        pares["target"] * pares[f"w{semana}"]
    )

    pares[f"act_w{semana}"] = (
        pares["actividad_actual"] * pares[f"w{semana}"]
    )

    pares[f"act_w{semana}_target"] = (
        pares["actividad_actual"]
        * pares[f"w{semana}"]
        * pares["target"]
    )

pares["act_target"] = (
    pares["actividad_actual"] * pares["target"]
)

# =====================================================
# 7. ESTIMAR MODELO
# =====================================================

formula = """
actividad_siguiente ~
actividad_actual
+ target
+ w1 + w2 + w3
+ target_w1 + target_w2 + target_w3
+ act_target
+ act_w1 + act_w2 + act_w3
+ act_w1_target + act_w2_target + act_w3_target
+ C(grupo)
"""

modelo = smf.ols(
    formula,
    data=pares
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": pares["id_student"]
    }
)

print("\nRESUMEN DEL MODELO")
print(modelo.summary())

parametros = modelo.params
covarianza = modelo.cov_params()

# =====================================================
# 8. CONSTRUIR COEFICIENTES DE PERSISTENCIA
# =====================================================

vectores = {}

for semana in range(4):

    contraste = pd.Series(
        0.0,
        index=parametros.index
    )

    # Persistencia del grupo target = 0.
    contraste["actividad_actual"] = 1.0

    if semana > 0:

        contraste[f"act_w{semana}"] = 1.0

    vectores[(0, semana)] = contraste

    # Diferencia adicional de persistencia
    # para el grupo target = 1.
    contraste_target = contraste.copy()

    contraste_target["act_target"] = 1.0

    if semana > 0:

        contraste_target[
            f"act_w{semana}_target"
        ] = 1.0

    vectores[(1, semana)] = contraste_target

# =====================================================
# 9. ESTIMAR PERSISTENCIA POR GRUPO Y SEMANA
# =====================================================

resultados = []

for target in [0, 1]:

    nombre_grupo = (
        "No abandono futuro"
        if target == 0
        else "Abandono futuro"
    )

    for semana in range(4):

        contraste = vectores[(target, semana)]

        rho = float(
            contraste @ parametros
        )

        varianza = float(
            contraste @ covarianza @ contraste
        )

        error = np.sqrt(max(varianza, 0))

        z = rho / error

        p = 2 * norm.sf(abs(z))

        resultados.append({
            "grupo_futuro": nombre_grupo,
            "target": target,
            "transicion": f"{semana} a {semana + 1}",
            "rho": rho,
            "error_estandar": error,
            "ic95_inferior": rho - 1.96 * error,
            "ic95_superior": rho + 1.96 * error,
            "p_valor": p
        })

tabla_persistencia = pd.DataFrame(resultados)

# =====================================================
# 10. COMPARAR LOS DOS GRUPOS EN CADA TRANSICION
# =====================================================

resultados_diferencias = []

for semana in range(4):

    contraste = (
        vectores[(1, semana)]
        - vectores[(0, semana)]
    )

    diferencia = float(
        contraste @ parametros
    )

    varianza = float(
        contraste @ covarianza @ contraste
    )

    error = np.sqrt(max(varianza, 0))

    z = diferencia / error

    p = 2 * norm.sf(abs(z))

    resultados_diferencias.append({
        "transicion": f"{semana} a {semana + 1}",
        "diferencia_rho": diferencia,
        "error_estandar": error,
        "ic95_inferior": diferencia - 1.96 * error,
        "ic95_superior": diferencia + 1.96 * error,
        "p_valor": p
    })

tabla_diferencias = pd.DataFrame(
    resultados_diferencias
)

# =====================================================
# 11. PRUEBA GLOBAL DE DIFERENCIAS
# =====================================================

# H0: no existen diferencias de persistencia
# entre target 0 y target 1 en ninguna transición.

matriz = np.vstack([
    (
        vectores[(1, semana)]
        - vectores[(0, semana)]
    ).values
    for semana in range(4)
])

diferencias = matriz @ parametros.values

cov_global = (
    matriz
    @ covarianza.values
    @ matriz.T
)

estadistico = float(
    diferencias.T
    @ np.linalg.pinv(cov_global)
    @ diferencias
)

p_global = chi2.sf(
    estadistico,
    df=4
)

tabla_global = pd.DataFrame([{
    "chi2_global": estadistico,
    "grados_libertad": 4,
    "p_global": p_global
}])

# =====================================================
# 12. MOSTRAR RESULTADOS
# =====================================================

print("\n===================================")
print("PERSISTENCIA POR GRUPO")
print("===================================")

print(
    tabla_persistencia.to_string(index=False)
)

print("\n===================================")
print("DIFERENCIAS ENTRE GRUPOS")
print("===================================")

print(
    tabla_diferencias.to_string(index=False)
)

print("\n===================================")
print("PRUEBA GLOBAL DE DIFERENCIAS")
print("===================================")

print(
    tabla_global.to_string(index=False)
)

# =====================================================
# 13. GUARDAR RESULTADOS
# =====================================================

tabla_persistencia.to_csv(
    CARPETA_SALIDA /
    "140_persistencia_por_grupo.csv",
    index=False
)

tabla_diferencias.to_csv(
    CARPETA_SALIDA /
    "140_diferencias_entre_grupos.csv",
    index=False
)

tabla_global.to_csv(
    CARPETA_SALIDA /
    "140_prueba_global.csv",
    index=False
)

print("\nARCHIVOS GUARDADOS EN:")
print(CARPETA_SALIDA)

print("\nEXPERIMENTO 140 FINALIZADO")
