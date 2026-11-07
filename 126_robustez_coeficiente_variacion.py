# 126_robustez_coeficiente_variacion.py

import pandas as pd
import numpy as np
import statsmodels.api as sm

from pathlib import Path
from scipy.stats import norm

# =====================================================
# 1. RUTAS
# =====================================================

BASE = Path(__file__).resolve().parent.parent

ARCHIVO = BASE / "data" / "interim" / "weekly_panel_weeks0_4.parquet"

SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. CARGAR DATOS
# =====================================================

df = pd.read_parquet(ARCHIVO)

print("DATOS CARGADOS")
print("============================")
print("Registros iniciales:", len(df))

# =====================================================
# 3. CALCULAR MEDIDAS DE ACTIVIDAD
# =====================================================

columnas_clicks = [
    "clics_semana_0",
    "clics_semana_1",
    "clics_semana_2",
    "clics_semana_3",
    "clics_semana_4"
]

df["semanas_observadas"] = df[columnas_clicks].notna().sum(axis=1)

df["media_actividad"] = df[columnas_clicks].mean(axis=1)

df["desviacion_estandar"] = df[columnas_clicks].std(
    axis=1,
    ddof=1
)

df["cv"] = (
    df["desviacion_estandar"] /
    df["media_actividad"]
)

# =====================================================
# 4. SELECCIONAR MUESTRA ORIGINAL
# =====================================================

df = df[
    (df["semanas_observadas"] >= 2) &
    (df["media_actividad"] > 0) &
    (df["desviacion_estandar"].notna()) &
    (df["cv"].notna())
].copy()

print("\nMUESTRA ORIGINAL")
print("============================")
print("Registros:", len(df))

# Umbrales calculados sobre la muestra original
limite_cv = df["cv"].quantile(0.99)

mediana_media = df["media_actividad"].median()

print("Percentil 99 del CV:", round(limite_cv, 5))
print("Mediana de actividad media:", round(mediana_media, 5))

# =====================================================
# 5. CREAR LAS TRES MUESTRAS
# =====================================================

muestras = {
    "Original": df.copy(),

    "Sin_CV_extremo": df[
        df["cv"] <= limite_cv
    ].copy(),

    "Actividad_media_alta": df[
        df["media_actividad"] >= mediana_media
    ].copy()
}

# =====================================================
# 6. FUNCION PARA ESTANDARIZAR
# =====================================================

def estandarizar(serie):
    desviacion = serie.std()

    if desviacion == 0:
        return serie * 0

    return (serie - serie.mean()) / desviacion

# =====================================================
# 7. FUNCION PARA AJUSTAR EL MODELO
# =====================================================

def ajustar_modelo(datos, nombre):

    datos = datos.copy()

    # Estandarizar variables dentro de cada muestra
    datos["media_z"] = estandarizar(
        datos["media_actividad"]
    )

    datos["sd_z"] = estandarizar(
        datos["desviacion_estandar"]
    )

    datos["cv_z"] = estandarizar(
        datos["cv"]
    )

    datos["semanas_z"] = estandarizar(
        datos["semanas_observadas"]
    )

    # Interacciones
    datos["interaccion_sd"] = (
        datos["sd_z"] * datos["semanas_z"]
    )

    datos["interaccion_cv"] = (
        datos["cv_z"] * datos["semanas_z"]
    )

    # Identificador académico
    datos["grupo"] = (
        datos["code_module"].astype(str) + "_" +
        datos["code_presentation"].astype(str)
    )

    # Variable objetivo
    y = datos["target_future_withdrawal"].astype(int)

    # Efectos fijos
    dummies = pd.get_dummies(
        datos["grupo"],
        prefix="grupo",
        drop_first=True,
        dtype=int
    )

    # Predictores del modelo conjunto
    X = datos[
        [
            "media_z",
            "sd_z",
            "cv_z",
            "semanas_z",
            "interaccion_sd",
            "interaccion_cv"
        ]
    ].copy()

    X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)

    # Ajustar regresión logística con HC3
    modelo = sm.Logit(y, X).fit(
        disp=False,
        maxiter=200,
        cov_type="HC3"
    )

    print("\n========================================")
    print("MODELO:", nombre)
    print("========================================")

    print("Registros:", len(datos))
    print("Grupos académicos:", datos["grupo"].nunique())
    print("Pseudo R2:", round(modelo.prsquared, 6))
    print("AIC:", round(modelo.aic, 3))
    print("BIC:", round(modelo.bic, 3))

    print("\nCOEFICIENTES PRINCIPALES")

    print(
        pd.DataFrame({
            "coeficiente": modelo.params,
            "error_HC3": modelo.bse,
            "valor_p": modelo.pvalues
        }).loc[
            [
                "media_z",
                "sd_z",
                "cv_z",
                "semanas_z",
                "interaccion_sd",
                "interaccion_cv"
            ]
        ].round(6)
    )

    # =============================================
    # EFECTOS CONDICIONALES DEL CV
    # =============================================

    efectos = []

    for semanas in [2, 3, 4, 5]:

        semanas_z = (
            semanas - datos["semanas_observadas"].mean()
        ) / datos["semanas_observadas"].std()

        coef = (
            modelo.params["cv_z"] +
            modelo.params["interaccion_cv"] * semanas_z
        )

        cov = modelo.cov_params()

        varianza = (
            cov.loc["cv_z", "cv_z"]
            + (semanas_z ** 2) *
            cov.loc["interaccion_cv", "interaccion_cv"]
            + 2 * semanas_z *
            cov.loc["cv_z", "interaccion_cv"]
        )

        error = np.sqrt(varianza)

        z = coef / error

        p = 2 * norm.sf(abs(z))

        limite_inf = coef - 1.96 * error
        limite_sup = coef + 1.96 * error

        efectos.append({
            "muestra": nombre,
            "semanas": semanas,
            "coeficiente_CV": coef,
            "error_estandar": error,
            "valor_p": p,
            "OR": np.exp(coef),
            "IC95_inferior_OR": np.exp(limite_inf),
            "IC95_superior_OR": np.exp(limite_sup)
        })

    return modelo, efectos

# =====================================================
# 8. EJECUTAR LOS TRES MODELOS
# =====================================================

resultados_modelos = []
resultados_efectos = []

for nombre, muestra in muestras.items():

    modelo, efectos = ajustar_modelo(
        muestra,
        nombre
    )

    resultados_modelos.append({
        "muestra": nombre,
        "registros": int(modelo.nobs),
        "grupos_academicos": muestra[
            ["code_module", "code_presentation"]
        ].drop_duplicates().shape[0],
        "pseudo_r2": modelo.prsquared,
        "log_likelihood": modelo.llf,
        "AIC": modelo.aic,
        "BIC": modelo.bic,
        "coeficiente_CV": modelo.params["cv_z"],
        "p_CV": modelo.pvalues["cv_z"],
        "coeficiente_interaccion_CV": modelo.params[
            "interaccion_cv"
        ],
        "p_interaccion_CV": modelo.pvalues[
            "interaccion_cv"
        ]
    })

    resultados_efectos.extend(efectos)

# =====================================================
# 9. TABLAS COMPARATIVAS
# =====================================================

tabla_modelos = pd.DataFrame(resultados_modelos)

tabla_efectos = pd.DataFrame(resultados_efectos)

print("\nCOMPARACION DE LOS TRES MODELOS")
print("============================")

print(
    tabla_modelos[
        [
            "muestra",
            "registros",
            "pseudo_r2",
            "AIC",
            "BIC",
            "coeficiente_CV",
            "p_CV",
            "coeficiente_interaccion_CV",
            "p_interaccion_CV"
        ]
    ].round(6).to_string(index=False)
)

print("\nEFECTOS CONDICIONALES DEL CV")
print("============================")

print(
    tabla_efectos.round(6).to_string(index=False)
)

# =====================================================
# 10. GUARDAR RESULTADOS
# =====================================================

tabla_modelos.to_csv(
    SALIDA / "126_comparacion_modelos.csv",
    index=False
)

tabla_efectos.to_csv(
    SALIDA / "126_efectos_condicionales_cv.csv",
    index=False
)

print("\nARCHIVOS GUARDADOS")
print("============================")
print("126_comparacion_modelos.csv")
print("126_efectos_condicionales_cv.csv")

print("\nANALISIS FINALIZADO")