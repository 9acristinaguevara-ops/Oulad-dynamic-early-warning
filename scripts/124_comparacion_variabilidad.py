# 124_comparacion_variabilidad.py

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
# 3. CALCULAR ACTIVIDAD SEMANAL
# =====================================================

columnas_clicks = [
    "clics_semana_0",
    "clics_semana_1",
    "clics_semana_2",
    "clics_semana_3",
    "clics_semana_4"
]

# Número de semanas con actividad observada
df["semanas_observadas"] = df[columnas_clicks].notna().sum(axis=1)

# Media de clics observados
df["media_actividad"] = df[columnas_clicks].mean(axis=1)

# Desviación estándar de los clics observados
df["desviacion_estandar"] = df[columnas_clicks].std(axis=1, ddof=1)

# Coeficiente de variación
df["cv"] = df["desviacion_estandar"] / df["media_actividad"]

# =====================================================
# 4. SELECCIONAR LA MISMA MUESTRA
# =====================================================

df = df[
    (df["semanas_observadas"] >= 2) &
    (df["media_actividad"] > 0) &
    (df["desviacion_estandar"].notna()) &
    (df["cv"].notna())
].copy()

print("\nMUESTRA DE ANALISIS")
print("============================")
print("Registros:", len(df))
print("Semanas observadas:")
print(df["semanas_observadas"].value_counts().sort_index())

# =====================================================
# 5. CREAR GRUPOS ACADEMICOS
# =====================================================

df["grupo"] = (
    df["code_module"].astype(str) + "_" +
    df["code_presentation"].astype(str)
)

print("\nGrupos académicos:", df["grupo"].nunique())

# =====================================================
# 6. ESTANDARIZAR VARIABLES
# =====================================================

def estandarizar(serie):
    return (serie - serie.mean()) / serie.std()

df["media_z"] = estandarizar(df["media_actividad"])

df["sd_z"] = estandarizar(df["desviacion_estandar"])

df["cv_z"] = estandarizar(df["cv"])

df["semanas_z"] = estandarizar(df["semanas_observadas"])

# Interacciones
df["interaccion_sd"] = df["sd_z"] * df["semanas_z"]

df["interaccion_cv"] = df["cv_z"] * df["semanas_z"]

# =====================================================
# 7. VARIABLE OBJETIVO Y EFECTOS FIJOS
# =====================================================

y = df["target_future_withdrawal"].astype(int)

# Variables indicadoras para los grupos académicos
dummies = pd.get_dummies(
    df["grupo"],
    prefix="grupo",
    drop_first=True,
    dtype=int
)

# =====================================================
# 8. MODELO A: DESVIACION ESTANDAR
# =====================================================

X_sd = df[
    [
        "media_z",
        "sd_z",
        "semanas_z",
        "interaccion_sd"
    ]
].copy()

X_sd = pd.concat([X_sd, dummies], axis=1)

X_sd = sm.add_constant(X_sd)

modelo_sd = sm.Logit(y, X_sd).fit(
    disp=False,
    cov_type="HC3"
)

# =====================================================
# 9. MODELO B: COEFICIENTE DE VARIACION
# =====================================================

X_cv = df[
    [
        "media_z",
        "cv_z",
        "semanas_z",
        "interaccion_cv"
    ]
].copy()

X_cv = pd.concat([X_cv, dummies], axis=1)

X_cv = sm.add_constant(X_cv)

modelo_cv = sm.Logit(y, X_cv).fit(
    disp=False,
    cov_type="HC3"
)

# =====================================================
# 10. COMPARACION GENERAL
# =====================================================

print("\nCOMPARACION DE MODELOS")
print("============================")

print("\nMODELO A: DESVIACION ESTANDAR")
print("Pseudo R2:", round(modelo_sd.prsquared, 6))
print("Log-Likelihood:", round(modelo_sd.llf, 3))
print("AIC:", round(modelo_sd.aic, 3))
print("BIC:", round(modelo_sd.bic, 3))

print("\nMODELO B: COEFICIENTE DE VARIACION")
print("Pseudo R2:", round(modelo_cv.prsquared, 6))
print("Log-Likelihood:", round(modelo_cv.llf, 3))
print("AIC:", round(modelo_cv.aic, 3))
print("BIC:", round(modelo_cv.bic, 3))

# =====================================================
# 11. EFECTOS CONDICIONALES
# =====================================================

def calcular_efectos(modelo, variable, interaccion, nombre):

    print("\nEFECTOS CONDICIONALES:", nombre)
    print("============================")

    for semanas in [2, 3, 4, 5]:

        semanas_z = (
            semanas - df["semanas_observadas"].mean()
        ) / df["semanas_observadas"].std()

        coef = (
            modelo.params[variable] +
            modelo.params[interaccion] * semanas_z
        )

        cov = modelo.cov_params()

        varianza = (
            cov.loc[variable, variable]
            + (semanas_z ** 2) * cov.loc[interaccion, interaccion]
            + 2 * semanas_z * cov.loc[variable, interaccion]
        )

        error = np.sqrt(varianza)

        z = coef / error

        p = 2 * (1 - norm.cdf(abs(z)))

        limite_inf = coef - 1.96 * error
        limite_sup = coef + 1.96 * error

        print("\nSemanas:", semanas)
        print("Coeficiente:", round(coef, 5))
        print("Error estándar:", round(error, 5))
        print("Valor p:", f"{p:.6g}")
        print("OR:", round(np.exp(coef), 5))
        print(
            "IC 95% OR:",
            round(np.exp(limite_inf), 5),
            "a",
            round(np.exp(limite_sup), 5)
        )


calcular_efectos(
    modelo_sd,
    "sd_z",
    "interaccion_sd",
    "DESVIACION ESTANDAR"
)

calcular_efectos(
    modelo_cv,
    "cv_z",
    "interaccion_cv",
    "COEFICIENTE DE VARIACION"
)

# =====================================================
# 12. GUARDAR RESULTADOS
# =====================================================

comparacion = pd.DataFrame([
    {
        "modelo": "Desviacion_estandar",
        "n": len(df),
        "pseudo_r2": modelo_sd.prsquared,
        "log_likelihood": modelo_sd.llf,
        "AIC": modelo_sd.aic,
        "BIC": modelo_sd.bic
    },
    {
        "modelo": "Coeficiente_variacion",
        "n": len(df),
        "pseudo_r2": modelo_cv.prsquared,
        "log_likelihood": modelo_cv.llf,
        "AIC": modelo_cv.aic,
        "BIC": modelo_cv.bic
    }
])

ruta_salida = SALIDA / "124_comparacion_variabilidad.csv"

comparacion.to_csv(ruta_salida, index=False)

print("\nARCHIVO GUARDADO EN:")
print(ruta_salida)

print("\nANALISIS FINALIZADO")