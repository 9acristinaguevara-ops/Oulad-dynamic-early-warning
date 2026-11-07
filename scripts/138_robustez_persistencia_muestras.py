
# 138_robustez_persistencia_muestras.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from math import erf

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

claves = [
    "code_module",
    "code_presentation",
    "id_student"
]

# =====================================================
# 3. CONSTRUIR FORMATO LONGITUDINAL
# =====================================================

registros = []

for semana in range(5):

    columna = f"clics_semana_{semana}"

    temporal = df[
        claves + [columna]
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

# Mantener únicamente semanas con registros disponibles.
longitudinal = longitudinal.dropna(
    subset=["clics"]
).copy()

# Transformación logarítmica.
longitudinal["log_clics"] = np.log1p(
    longitudinal["clics"]
)

# Identificar módulo y presentación.
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
# 5. CONSTRUIR TRANSICIONES CONSECUTIVAS
# =====================================================

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
print("Observaciones:", len(pares))
print("Estudiantes:", pares["id_student"].nunique())

# =====================================================
# 6. CONTAR TRANSICIONES POR ESTUDIANTE-CURSO
# =====================================================

conteo = (
    pares.groupby(claves)
    .size()
    .reset_index(name="numero_transiciones")
)

pares = pares.merge(
    conteo,
    on=claves,
    how="left"
)

print("\nDISTRIBUCION DE TRANSICIONES")

print(
    conteo["numero_transiciones"]
    .value_counts()
    .sort_index()
)

# =====================================================
# 7. DEFINIR MUESTRAS
# =====================================================

criterios = {
    "Al menos 2 transiciones": 2,
    "Al menos 3 transiciones": 3,
    "4 transiciones completas": 4
}

resultados = []
resumen_muestras = []

# =====================================================
# 8. ESTIMAR MODELOS POR MUESTRA
# =====================================================

for nombre_muestra, minimo in criterios.items():

    muestra = pares[
        pares["numero_transiciones"] >= minimo
    ].copy()

    print("\n===================================")
    print(nombre_muestra)
    print("===================================")

    print("Observaciones:", len(muestra))
    print("Estudiantes:", muestra["id_student"].nunique())
    print(
        "Estudiantes-curso:",
        muestra[claves].drop_duplicates().shape[0]
    )

    resumen_muestras.append({
        "muestra": nombre_muestra,
        "minimo_transiciones": minimo,
        "observaciones": len(muestra),
        "estudiantes": muestra["id_student"].nunique(),
        "estudiantes_curso": muestra[claves].drop_duplicates().shape[0]
    })

    # Modelo con persistencia variable según la semana.

    modelo = smf.ols(
        """
        actividad_siguiente ~
        actividad_actual * C(semana)
        + C(grupo)
        """,
        data=muestra
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": muestra["id_student"]
        }
    )

    parametros = modelo.params
    covarianza = modelo.cov_params()

    # Extraer persistencia de cada transición.

    for semana in range(4):

        contraste = pd.Series(
            0.0,
            index=parametros.index
        )

        contraste["actividad_actual"] = 1.0

        if semana > 0:

            nombre_interaccion = (
                f"actividad_actual:C(semana)[T.{semana}]"
            )

            contraste[nombre_interaccion] = 1.0

        rho = float(
            contraste @ parametros
        )

        varianza = float(
            contraste @ covarianza @ contraste
        )

        error = np.sqrt(max(varianza, 0))

        z = rho / error

        p_valor = 2 * (
                1 - 0.5 * (
                1 + erf(abs(z) / np.sqrt(2))
        )
        )
        resultados.append({
            "muestra": nombre_muestra,
            "transicion": f"{semana} a {semana + 1}",
            "persistencia_rho": rho,
            "error_estandar": error,
            "ic95_inferior": rho - 1.96 * error,
            "ic95_superior": rho + 1.96 * error,
            "p_valor": p_valor,
            "r_squared": modelo.rsquared
        })

# =====================================================
# 9. TABLAS DE RESULTADOS
# =====================================================

tabla_resultados = pd.DataFrame(resultados)

tabla_muestras = pd.DataFrame(
    resumen_muestras
)

print("\nRESULTADOS DE PERSISTENCIA")

print(
    tabla_resultados.to_string(index=False)
)

print("\nRESUMEN DE MUESTRAS")

print(
    tabla_muestras.to_string(index=False)
)

# =====================================================
# 10. GUARDAR ARCHIVOS
# =====================================================

tabla_resultados.to_csv(
    CARPETA_SALIDA /
    "138_robustez_persistencia.csv",
    index=False
)

tabla_muestras.to_csv(
    CARPETA_SALIDA /
    "138_resumen_muestras.csv",
    index=False
)

print("\nARCHIVOS GUARDADOS EN:")
print(CARPETA_SALIDA)

print("\nEXPERIMENTO 138 FINALIZADO")
