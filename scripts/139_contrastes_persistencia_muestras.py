
# 139_contrastes_persistencia_muestras.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from pathlib import Path
from scipy.stats import norm, chi2
from statsmodels.stats.multitest import multipletests

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

claves = [
    "code_module",
    "code_presentation",
    "id_student"
]

print("DIMENSIONES DEL PANEL")
print(df.shape)

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

# =====================================================
# 7. DEFINIR MUESTRAS
# =====================================================

criterios = {
    "Al menos 2 transiciones": 2,
    "Al menos 3 transiciones": 3,
    "4 transiciones completas": 4
}

resultados_globales = []
resultados_pares = []
resumen_muestras = []

# =====================================================
# 8. ESTIMAR MODELOS Y CONTRASTES
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

    # Modelo de persistencia variable por transición.

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

    # Construir los cuatro coeficientes de persistencia.

    vectores = []

    for semana in range(4):

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

        vectores.append(contraste)

    coeficientes = []

    for semana, contraste in enumerate(vectores):

        rho = float(contraste @ parametros)

        varianza = float(
            contraste @ covarianza @ contraste
        )

        error = np.sqrt(max(varianza, 0))

        z = rho / error

        p = 2 * norm.sf(abs(z))

        inferior = rho - 1.96 * error
        superior = rho + 1.96 * error

        coeficientes.append(rho)

        resultados_globales.append({
            "muestra": nombre_muestra,
            "transicion": f"{semana} a {semana + 1}",
            "rho": rho,
            "error_estandar": error,
            "ic95_inferior": inferior,
            "ic95_superior": superior,
            "p_valor": p
        })

    # -------------------------------------------------
    # 8.1. PRUEBA GLOBAL DE IGUALDAD
    # -------------------------------------------------

    # H0: los cuatro coeficientes de persistencia
    # son iguales entre sí.

    matriz_global = np.vstack([
        (vectores[1] - vectores[0]).values,
        (vectores[2] - vectores[0]).values,
        (vectores[3] - vectores[0]).values
    ])

    diferencias_globales = (
        matriz_global @ parametros.values
    )

    cov_global = (
        matriz_global
        @ covarianza.values
        @ matriz_global.T
    )

    estadistico = float(
        diferencias_globales.T
        @ np.linalg.pinv(cov_global)
        @ diferencias_globales
    )

    p_global = chi2.sf(
        estadistico,
        df=3
    )

    resultados_muestra = {
        "muestra": nombre_muestra,
        "observaciones": len(muestra),
        "estudiantes": muestra["id_student"].nunique(),
        "chi2_global": estadistico,
        "grados_libertad": 3,
        "p_global": p_global,
        "r_squared": modelo.rsquared
    }

    resumen_muestras.append(resultados_muestra)

    # -------------------------------------------------
    # 8.2. SEIS COMPARACIONES POR PARES
    # -------------------------------------------------

    comparaciones = []

    for i in range(4):

        for j in range(i + 1, 4):

            contraste = vectores[i] - vectores[j]

            diferencia = float(
                contraste @ parametros
            )

            varianza = float(
                contraste @ covarianza @ contraste
            )

            error = np.sqrt(max(varianza, 0))

            z = diferencia / error

            p = 2 * norm.sf(abs(z))

            comparaciones.append({
                "muestra": nombre_muestra,
                "comparacion": (
                    f"{i}->{i+1} vs {j}->{j+1}"
                ),
                "diferencia": diferencia,
                "error_estandar": error,
                "ic95_inferior": diferencia - 1.96 * error,
                "ic95_superior": diferencia + 1.96 * error,
                "p_sin_ajustar": p
            })

    # Corrección Holm para seis comparaciones.

    p_originales = [
        resultado["p_sin_ajustar"]
        for resultado in comparaciones
    ]

    rechazados, p_ajustados, _, _ = multipletests(
        p_originales,
        alpha=0.05,
        method="holm"
    )

    for k in range(len(comparaciones)):

        comparaciones[k]["p_holm"] = p_ajustados[k]
        comparaciones[k]["significativo_holm"] = rechazados[k]

        resultados_pares.append(
            comparaciones[k]
        )

# =====================================================
# 9. CREAR TABLAS
# =====================================================

tabla_global = pd.DataFrame(
    resumen_muestras
)

tabla_coeficientes = pd.DataFrame(
    resultados_globales
)

tabla_pares = pd.DataFrame(
    resultados_pares
)

# =====================================================
# 10. MOSTRAR RESULTADOS
# =====================================================

print("\n===================================")
print("PRUEBAS GLOBALES DE IGUALDAD")
print("===================================")

print(
    tabla_global.to_string(index=False)
)

print("\n===================================")
print("COEFICIENTES DE PERSISTENCIA")
print("===================================")

print(
    tabla_coeficientes.to_string(index=False)
)

print("\n===================================")
print("COMPARACIONES POR PARES - HOLM")
print("===================================")

print(
    tabla_pares.to_string(index=False)
)

# =====================================================
# 11. GUARDAR RESULTADOS
# =====================================================

tabla_global.to_csv(
    CARPETA_SALIDA / "139_pruebas_globales.csv",
    index=False
)

tabla_coeficientes.to_csv(
    CARPETA_SALIDA / "139_coeficientes_persistencia.csv",
    index=False
)

tabla_pares.to_csv(
    CARPETA_SALIDA / "139_comparaciones_holm.csv",
    index=False
)

print("\nARCHIVOS GUARDADOS EN:")
print(CARPETA_SALIDA)

print("\nEXPERIMENTO 139 FINALIZADO")
