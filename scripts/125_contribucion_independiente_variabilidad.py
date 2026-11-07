# 125_contribucion_independiente_variabilidad.py

import pandas as pd
import numpy as np
import statsmodels.api as sm

from pathlib import Path
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
# 3. CALCULAR MEDIDAS DE VARIABILIDAD
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

df["desviacion_estandar"] = df[columnas_clicks].std(axis=1, ddof=1)

df["cv"] = (
    df["desviacion_estandar"] /
    df["media_actividad"]
)

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
print("\nDistribución de semanas:")
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

# Interacciones con semanas observadas
df["interaccion_sd"] = df["sd_z"] * df["semanas_z"]

df["interaccion_cv"] = df["cv_z"] * df["semanas_z"]

# =====================================================
# 7. CORRELACIONES ENTRE VARIABLES
# =====================================================

variables_correlacion = [
    "media_z",
    "sd_z",
    "cv_z",
    "semanas_z",
    "interaccion_sd",
    "interaccion_cv"
]

correlaciones = df[variables_correlacion].corr()

print("\nMATRIZ DE CORRELACIONES")
print("============================")
print(correlaciones.round(3))

correlaciones.to_csv(
    SALIDA / "125_matriz_correlaciones.csv"
)

# =====================================================
# 8. DIAGNOSTICO DE MULTICOLINEALIDAD
# =====================================================

X_vif = df[variables_correlacion].copy()

X_vif = sm.add_constant(X_vif)

resultados_vif = []

for i, variable in enumerate(X_vif.columns):

    if variable == "const":
        continue

    vif = variance_inflation_factor(
        X_vif.values,
        i
    )

    resultados_vif.append({
        "variable": variable,
        "VIF": vif
    })

tabla_vif = pd.DataFrame(resultados_vif)

print("\nFACTOR DE INFLACION DE VARIANZA")
print("============================")
print(tabla_vif.round(3))

tabla_vif.to_csv(
    SALIDA / "125_diagnostico_vif.csv",
    index=False
)

# =====================================================
# 9. VARIABLE OBJETIVO Y EFECTOS FIJOS
# =====================================================

y = df["target_future_withdrawal"].astype(int)

dummies = pd.get_dummies(
    df["grupo"],
    prefix="grupo",
    drop_first=True,
    dtype=int
)

# =====================================================
# 10. MODELO CONJUNTO
# =====================================================

X = df[
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

modelo = sm.Logit(y, X).fit(
    disp=False,
    maxiter=200,
    cov_type="HC3"
)

print("\nRESULTADOS DEL MODELO CONJUNTO")
print("============================")
print(modelo.summary())

print("\nINDICADORES GENERALES")
print("============================")
print("Registros:", int(modelo.nobs))
print("Pseudo R2:", round(modelo.prsquared, 6))
print("Log-Likelihood:", round(modelo.llf, 3))
print("AIC:", round(modelo.aic, 3))
print("BIC:", round(modelo.bic, 3))

# =====================================================
# 11. EFECTOS CONDICIONALES
# =====================================================

def calcular_efecto(modelo, variable, interaccion, semanas):

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
        + (semanas_z ** 2) *
        cov.loc[interaccion, interaccion]
        + 2 * semanas_z *
        cov.loc[variable, interaccion]
    )

    error = np.sqrt(varianza)

    z = coef / error

    p = 2 * norm.sf(abs(z))

    limite_inf = coef - 1.96 * error
    limite_sup = coef + 1.96 * error

    return {
        "semanas": semanas,
        "coeficiente": coef,
        "error_estandar": error,
        "valor_p": p,
        "OR": np.exp(coef),
        "IC95_inferior_OR": np.exp(limite_inf),
        "IC95_superior_OR": np.exp(limite_sup)
    }


resultados_efectos = []

for semanas in [2, 3, 4, 5]:

    efecto_sd = calcular_efecto(
        modelo,
        "sd_z",
        "interaccion_sd",
        semanas
    )

    efecto_sd["variable"] = "Desviacion_estandar"

    resultados_efectos.append(efecto_sd)

    efecto_cv = calcular_efecto(
        modelo,
        "cv_z",
        "interaccion_cv",
        semanas
    )

    efecto_cv["variable"] = "Coeficiente_variacion"

    resultados_efectos.append(efecto_cv)


tabla_efectos = pd.DataFrame(resultados_efectos)

print("\nEFECTOS CONDICIONALES DEL MODELO CONJUNTO")
print("============================")
print(
    tabla_efectos[
        [
            "variable",
            "semanas",
            "coeficiente",
            "error_estandar",
            "valor_p",
            "OR",
            "IC95_inferior_OR",
            "IC95_superior_OR"
        ]
    ].round(6).to_string(index=False)
)

# =====================================================
# 12. GUARDAR COEFICIENTES DEL MODELO
# =====================================================

tabla_coeficientes = pd.DataFrame({
    "variable": modelo.params.index,
    "coeficiente": modelo.params.values,
    "error_estandar_HC3": modelo.bse.values,
    "valor_p": modelo.pvalues.values,
    "OR": np.exp(modelo.params.values)
})

tabla_coeficientes.to_csv(
    SALIDA / "125_coeficientes_modelo_conjunto.csv",
    index=False
)

tabla_efectos.to_csv(
    SALIDA / "125_efectos_condicionales.csv",
    index=False
)

# =====================================================
# 13. GUARDAR INDICADORES GENERALES
# =====================================================

indicadores = pd.DataFrame([{
    "registros": int(modelo.nobs),
    "grupos_academicos": df["grupo"].nunique(),
    "pseudo_r2": modelo.prsquared,
    "log_likelihood": modelo.llf,
    "AIC": modelo.aic,
    "BIC": modelo.bic
}])

indicadores.to_csv(
    SALIDA / "125_indicadores_modelo.csv",
    index=False
)

# =====================================================
# 14. FINALIZAR
# =====================================================

print("\nARCHIVOS GUARDADOS")
print("============================")
print("125_matriz_correlaciones.csv")
print("125_diagnostico_vif.csv")
print("125_coeficientes_modelo_conjunto.csv")
print("125_efectos_condicionales.csv")
print("125_indicadores_modelo.csv")

print("\nANALISIS FINALIZADO")