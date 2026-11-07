# 127_control_no_lineal_media.py

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
# 3. CALCULAR ACTIVIDAD Y VARIABILIDAD
# =====================================================

columnas_clicks = [
    "clics_semana_0",
    "clics_semana_1",
    "clics_semana_2",
    "clics_semana_3",
    "clics_semana_4"
]

df["semanas_observadas"] = (
    df[columnas_clicks].notna().sum(axis=1)
)

df["media_actividad"] = (
    df[columnas_clicks].mean(axis=1)
)

df["desviacion_estandar"] = (
    df[columnas_clicks].std(axis=1, ddof=1)
)

df["cv"] = (
    df["desviacion_estandar"] /
    df["media_actividad"]
)

# =====================================================
# 4. SELECCIONAR MUESTRA
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

# =====================================================
# 5. TRANSFORMACION NO LINEAL DE LA MEDIA
# =====================================================

# Transformación logarítmica
df["log_media"] = np.log1p(df["media_actividad"])

# Estandarizar logaritmo de la media
df["log_media_z"] = (
    df["log_media"] - df["log_media"].mean()
) / df["log_media"].std()

# Término cuadrático
df["log_media_z2"] = df["log_media_z"] ** 2

# Media lineal para el modelo A
df["media_z"] = (
    df["media_actividad"] -
    df["media_actividad"].mean()
) / df["media_actividad"].std()

# =====================================================
# 6. ESTANDARIZAR VARIABLES
# =====================================================

df["cv_z"] = (
    df["cv"] - df["cv"].mean()
) / df["cv"].std()

df["sd_z"] = (
    df["desviacion_estandar"] -
    df["desviacion_estandar"].mean()
) / df["desviacion_estandar"].std()

df["semanas_z"] = (
    df["semanas_observadas"] -
    df["semanas_observadas"].mean()
) / df["semanas_observadas"].std()

# Interacciones
df["interaccion_cv"] = (
    df["cv_z"] * df["semanas_z"]
)

df["interaccion_sd"] = (
    df["sd_z"] * df["semanas_z"]
)

# =====================================================
# 7. GRUPOS ACADEMICOS
# =====================================================

df["grupo"] = (
    df["code_module"].astype(str) + "_" +
    df["code_presentation"].astype(str)
)

dummies = pd.get_dummies(
    df["grupo"],
    prefix="grupo",
    drop_first=True,
    dtype=int
)

print("Grupos académicos:", df["grupo"].nunique())

# =====================================================
# 8. VARIABLE OBJETIVO
# =====================================================

y = df["target_future_withdrawal"].astype(int)

# =====================================================
# 9. FUNCION PARA AJUSTAR MODELOS
# =====================================================

def ajustar_modelo(nombre, variables):

    X = df[variables].copy()

    X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)

    modelo = sm.Logit(y, X).fit(
        disp=False,
        maxiter=200,
        cov_type="HC3"
    )

    print("\n========================================")
    print("MODELO:", nombre)
    print("========================================")

    print("Registros:", int(modelo.nobs))
    print("Pseudo R2:", round(modelo.prsquared, 6))
    print("Log-Likelihood:", round(modelo.llf, 3))
    print("AIC:", round(modelo.aic, 3))
    print("BIC:", round(modelo.bic, 3))

    print("\nCOEFICIENTES PRINCIPALES")

    principales = [
        variable for variable in variables
    ]

    print(
        pd.DataFrame({
            "coeficiente": modelo.params,
            "error_HC3": modelo.bse,
            "valor_p": modelo.pvalues
        }).loc[principales].round(6)
    )

    return modelo


# =====================================================
# 10. MODELO A: CONTROL LINEAL
# =====================================================

modelo_A = ajustar_modelo(
    "A - MEDIA LINEAL Y CV",
    [
        "media_z",
        "cv_z",
        "semanas_z",
        "interaccion_cv"
    ]
)

# =====================================================
# 11. MODELO B: CONTROL NO LINEAL
# =====================================================

modelo_B = ajustar_modelo(
    "B - MEDIA NO LINEAL Y CV",
    [
        "log_media_z",
        "log_media_z2",
        "cv_z",
        "semanas_z",
        "interaccion_cv"
    ]
)

# =====================================================
# 12. MODELO C: CONTROL NO LINEAL + SD + CV
# =====================================================

modelo_C = ajustar_modelo(
    "C - MEDIA NO LINEAL, SD Y CV",
    [
        "log_media_z",
        "log_media_z2",
        "sd_z",
        "cv_z",
        "semanas_z",
        "interaccion_sd",
        "interaccion_cv"
    ]
)

# =====================================================
# 13. COMPARACION GENERAL
# =====================================================

modelos = {
    "Modelo_A_lineal": modelo_A,
    "Modelo_B_no_lineal": modelo_B,
    "Modelo_C_no_lineal_SD_CV": modelo_C
}

resultados = []

for nombre, modelo in modelos.items():

    resultados.append({
        "modelo": nombre,
        "registros": int(modelo.nobs),
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

tabla_modelos = pd.DataFrame(resultados)

print("\nCOMPARACION DE MODELOS")
print("============================")

print(tabla_modelos.round(6).to_string(index=False))

# =====================================================
# 14. EFECTOS CONDICIONALES DEL CV
# =====================================================

def calcular_efectos(modelo, nombre):

    efectos = []

    for semanas in [2, 3, 4, 5]:

        semanas_z = (
            semanas - df["semanas_observadas"].mean()
        ) / df["semanas_observadas"].std()

        coef = (
            modelo.params["cv_z"] +
            modelo.params["interaccion_cv"] * semanas_z
        )

        cov = modelo.cov_params()

        varianza = (
            cov.loc["cv_z", "cv_z"]
            + semanas_z ** 2 *
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
            "modelo": nombre,
            "semanas": semanas,
            "coeficiente": coef,
            "error_estandar": error,
            "valor_p": p,
            "OR": np.exp(coef),
            "IC95_inferior_OR": np.exp(limite_inf),
            "IC95_superior_OR": np.exp(limite_sup)
        })

    return efectos


efectos = []

for nombre, modelo in modelos.items():

    efectos.extend(
        calcular_efectos(modelo, nombre)
    )

tabla_efectos = pd.DataFrame(efectos)

print("\nEFECTOS CONDICIONALES DEL CV")
print("============================")

print(tabla_efectos.round(6).to_string(index=False))

# =====================================================
# 15. GUARDAR RESULTADOS
# =====================================================

tabla_modelos.to_csv(
    SALIDA / "127_comparacion_modelos.csv",
    index=False
)

tabla_efectos.to_csv(
    SALIDA / "127_efectos_condicionales_cv.csv",
    index=False
)

print("\nARCHIVOS GUARDADOS")
print("============================")
print("127_comparacion_modelos.csv")
print("127_efectos_condicionales_cv.csv")

print("\nANALISIS FINALIZADO")