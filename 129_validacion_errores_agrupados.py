
# 129_validacion_errores_agrupados.py

import pandas as pd
import numpy as np
import statsmodels.api as sm

from pathlib import Path
from scipy.stats import t

# =====================================================
# 1. RUTAS
# =====================================================

BASE = Path(__file__).resolve().parent.parent

ARCHIVO = (
    BASE / "data" / "interim" /
    "weekly_panel_weeks0_4.parquet"
)

SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. CARGAR DATOS
# =====================================================

df = pd.read_parquet(ARCHIVO)

columnas_clicks = [
    "clics_semana_0",
    "clics_semana_1",
    "clics_semana_2",
    "clics_semana_3",
    "clics_semana_4"
]

# =====================================================
# 3. CONSTRUIR VARIABLES
# =====================================================

df["semanas_observadas"] = (
    df[columnas_clicks].notna().sum(axis=1)
)

df["media_actividad"] = (
    df[columnas_clicks].mean(axis=1)
)

df["sd"] = df[columnas_clicks].std(
    axis=1,
    ddof=1
)

df["cv"] = (
    df["sd"] / df["media_actividad"]
)

# =====================================================
# 4. SELECCIONAR LA MUESTRA
# =====================================================

df = df[
    (df["semanas_observadas"] >= 2) &
    (df["media_actividad"] > 0) &
    (df["cv"].notna())
].copy()

print("MUESTRA DE ANALISIS")
print("============================")
print("Registros:", len(df))

# =====================================================
# 5. VARIABLES DE CONTROL
# =====================================================

df["log_media"] = np.log1p(
    df["media_actividad"]
)

df["log_media_z"] = (
    df["log_media"] - df["log_media"].mean()
) / df["log_media"].std()

df["log_media_z2"] = (
    df["log_media_z"] ** 2
)

df["cv_z"] = (
    df["cv"] - df["cv"].mean()
) / df["cv"].std()

df["semanas_z"] = (
    df["semanas_observadas"] -
    df["semanas_observadas"].mean()
) / df["semanas_observadas"].std()

df["interaccion_cv"] = (
    df["cv_z"] * df["semanas_z"]
)

# =====================================================
# 6. IDENTIFICAR LOS GRUPOS ACADEMICOS
# =====================================================

df["grupo"] = (
    df["code_module"].astype(str) + "_" +
    df["code_presentation"].astype(str)
)

print("Grupos académicos:", df["grupo"].nunique())

# =====================================================
# 7. CONSTRUIR LA MATRIZ DEL MODELO
# =====================================================

variables = [
    "log_media_z",
    "log_media_z2",
    "cv_z",
    "semanas_z",
    "interaccion_cv"
]

X = df[variables].copy()

# Efectos fijos de los grupos académicos
dummies = pd.get_dummies(
    df["grupo"],
    prefix="grupo",
    drop_first=True,
    dtype=int
)

X = pd.concat([X, dummies], axis=1)

X = sm.add_constant(X)

y = df["target_future_withdrawal"].astype(int)

# =====================================================
# 8. MODELO CON ERRORES HC3
# =====================================================

modelo_hc3 = sm.Logit(y, X).fit(
    disp=False,
    maxiter=200,
    cov_type="HC3"
)

# =====================================================
# 9. MODELO CON ERRORES AGRUPADOS
# =====================================================

modelo_cluster = sm.Logit(y, X).fit(
    disp=False,
    maxiter=200,
    cov_type="cluster",
    cov_kwds={
        "groups": df["grupo"],
        "use_correction": True
    },
    use_t=True
)

# =====================================================
# 10. COMPARAR LOS COEFICIENTES
# =====================================================

variables_interes = [
    "log_media_z",
    "log_media_z2",
    "cv_z",
    "semanas_z",
    "interaccion_cv"
]

resultados = []

for variable in variables_interes:

    resultados.append({
        "variable": variable,

        "coeficiente_HC3":
            modelo_hc3.params[variable],

        "error_HC3":
            modelo_hc3.bse[variable],

        "p_HC3":
            modelo_hc3.pvalues[variable],

        "coeficiente_cluster":
            modelo_cluster.params[variable],

        "error_cluster":
            modelo_cluster.bse[variable],

        "p_cluster":
            modelo_cluster.pvalues[variable]
    })

tabla = pd.DataFrame(resultados)

print("\nCOMPARACION HC3 VS CLUSTER")
print("============================")
print(tabla.round(6).to_string(index=False))

# =====================================================
# 11. EFECTOS CONDICIONALES DEL CV
# =====================================================

efectos = []

for semanas in [2, 3, 4, 5]:

    semanas_z = (
        semanas - df["semanas_observadas"].mean()
    ) / df["semanas_observadas"].std()

    for nombre, modelo in [
        ("HC3", modelo_hc3),
        ("Cluster", modelo_cluster)
    ]:

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

        error = np.sqrt(max(varianza, 0))

        if nombre == "Cluster":
            grados_libertad = df["grupo"].nunique() - 1
            valor_p = 2 * t.sf(
                abs(coef / error),
                df=grados_libertad
            )
            critico = t.ppf(
                0.975,
                df=grados_libertad
            )
        else:
            from scipy.stats import norm
            valor_p = 2 * norm.sf(abs(coef / error))
            critico = 1.96

        efectos.append({
            "metodo": nombre,
            "semanas": semanas,
            "coeficiente": coef,
            "error_estandar": error,
            "valor_p": valor_p,
            "OR": np.exp(coef),
            "IC95_inferior_OR": np.exp(
                coef - critico * error
            ),
            "IC95_superior_OR": np.exp(
                coef + critico * error
            )
        })

tabla_efectos = pd.DataFrame(efectos)

print("\nEFECTOS CONDICIONALES DEL CV")
print("============================")
print(tabla_efectos.round(6).to_string(index=False))

# =====================================================
# 12. GUARDAR RESULTADOS
# =====================================================

tabla.to_csv(
    SALIDA / "129_comparacion_HC3_cluster.csv",
    index=False
)

tabla_efectos.to_csv(
    SALIDA / "129_efectos_condicionales.csv",
    index=False
)

print("\nARCHIVOS GUARDADOS")
print("============================")
print("129_comparacion_HC3_cluster.csv")
print("129_efectos_condicionales.csv")

print("\nANALISIS FINALIZADO")
