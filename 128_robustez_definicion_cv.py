
# 128_robustez_definicion_cv.py

import pandas as pd
import numpy as np
import statsmodels.api as sm

from pathlib import Path
from scipy.stats import norm

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

print("DATOS CARGADOS")
print("============================")
print("Registros iniciales:", len(df))

# =====================================================
# 3. CALCULAR ACTIVIDAD MEDIA Y SEMANAS OBSERVADAS
# =====================================================

df["semanas_observadas"] = (
    df[columnas_clicks].notna().sum(axis=1)
)

df["media_actividad"] = (
    df[columnas_clicks].mean(axis=1)
)

# Desviación estándar tradicional
df["sd"] = df[columnas_clicks].std(
    axis=1,
    ddof=1
)

# CV tradicional
df["cv_tradicional"] = (
    df["sd"] / df["media_actividad"]
)

# =====================================================
# 4. CALCULAR CV CON VALORES EXTREMOS LIMITADOS
# =====================================================

# Percentil 99 de todos los clics semanales observados
todos_los_clics = df[columnas_clicks].stack()

limite_p99 = todos_los_clics.quantile(0.99)

print("\nTRATAMIENTO DE VALORES EXTREMOS")
print("============================")
print("Percentil 99:", round(limite_p99, 3))

# Limitar los clics superiores al percentil 99
clics_limitados = df[columnas_clicks].clip(
    upper=limite_p99
)

# Calcular media y desviación estándar con datos limitados
media_limitada = clics_limitados.mean(axis=1)

sd_limitada = clics_limitados.std(
    axis=1,
    ddof=1
)

df["cv_limitado"] = (
    sd_limitada / media_limitada
)

# =====================================================
# 5. CALCULAR VARIABILIDAD RELATIVA ROBUSTA
# =====================================================

def calcular_cv_robusto(fila):

    valores = fila.dropna()

    mediana = valores.median()

    desviacion_absoluta = (
        valores - mediana
    ).abs().median()

    mad = 1.4826 * desviacion_absoluta

    if mediana <= 0:
        return np.nan

    return mad / mediana


df["cv_robusto"] = df[columnas_clicks].apply(
    calcular_cv_robusto,
    axis=1
)

# =====================================================
# 6. SELECCIONAR MUESTRA
# =====================================================

df = df[
    (df["semanas_observadas"] >= 2) &
    (df["media_actividad"] > 0) &
    (df["cv_tradicional"].notna()) &
    (df["cv_limitado"].notna()) &
    (df["cv_robusto"].notna())
].copy()

print("\nMUESTRA DE ANALISIS")
print("============================")
print("Registros:", len(df))
print("Grupos académicos:", df["code_module"].astype(str).add(
    "_"
).add(df["code_presentation"].astype(str)).nunique())

# =====================================================
# 7. DEFINIR VARIABLES DE CONTROL
# =====================================================

df["log_media"] = np.log1p(df["media_actividad"])

df["log_media_z"] = (
    df["log_media"] - df["log_media"].mean()
) / df["log_media"].std()

df["log_media_z2"] = df["log_media_z"] ** 2

# Variable de semanas observadas
df["semanas_z"] = (
    df["semanas_observadas"] -
    df["semanas_observadas"].mean()
) / df["semanas_observadas"].std()

# Grupos académicos
df["grupo"] = (
    df["code_module"].astype(str) + "_" +
    df["code_presentation"].astype(str)
)

# =====================================================
# 8. FUNCION PARA AJUSTAR LOS MODELOS
# =====================================================

def ajustar_modelo(datos, variable, nombre):

    datos = datos.copy()

    # Estandarizar indicador de variabilidad
    datos["variabilidad_z"] = (
        datos[variable] - datos[variable].mean()
    ) / datos[variable].std()

    # Interacción con semanas observadas
    datos["interaccion_var"] = (
        datos["variabilidad_z"] *
        datos["semanas_z"]
    )

    # Variables explicativas
    variables = [
        "log_media_z",
        "log_media_z2",
        "variabilidad_z",
        "semanas_z",
        "interaccion_var"
    ]

    X = datos[variables].copy()

    # Efectos fijos de grupos académicos
    dummies = pd.get_dummies(
        datos["grupo"],
        prefix="grupo",
        drop_first=True,
        dtype=int
    )

    X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)

    y = datos["target_future_withdrawal"].astype(int)

    modelo = sm.Logit(y, X).fit(
        disp=False,
        maxiter=200,
        cov_type="HC3"
    )

    print("\n========================================")
    print(nombre)
    print("========================================")

    print("Registros:", int(modelo.nobs))
    print("Pseudo R2:", round(modelo.prsquared, 6))
    print("AIC:", round(modelo.aic, 3))
    print("BIC:", round(modelo.bic, 3))

    print("\nCOEFICIENTES")

    print(
        pd.DataFrame({
            "coeficiente": modelo.params,
            "error_HC3": modelo.bse,
            "valor_p": modelo.pvalues
        }).loc[variables].round(6)
    )

    return modelo


# =====================================================
# 9. FUNCION PARA CALCULAR EFECTOS CONDICIONALES
# =====================================================

def calcular_efectos(modelo, datos, nombre):

    resultados = []

    for semanas in [2, 3, 4, 5]:

        semanas_z = (
            semanas - datos["semanas_observadas"].mean()
        ) / datos["semanas_observadas"].std()

        coef = (
            modelo.params["variabilidad_z"] +
            modelo.params["interaccion_var"] * semanas_z
        )

        cov = modelo.cov_params()

        varianza = (
            cov.loc["variabilidad_z", "variabilidad_z"]
            + semanas_z ** 2 *
            cov.loc["interaccion_var", "interaccion_var"]
            + 2 * semanas_z *
            cov.loc["variabilidad_z", "interaccion_var"]
        )

        error = np.sqrt(varianza)

        z = coef / error

        p = 2 * norm.sf(abs(z))

        resultados.append({
            "modelo": nombre,
            "semanas": semanas,
            "coeficiente": coef,
            "error_estandar": error,
            "valor_p": p,
            "OR": np.exp(coef),
            "IC95_inferior_OR": np.exp(coef - 1.96 * error),
            "IC95_superior_OR": np.exp(coef + 1.96 * error)
        })

    return resultados


# =====================================================
# 10. DEFINIR LOS INDICADORES
# =====================================================

indicadores = {
    "CV_tradicional": "cv_tradicional",
    "CV_limitado_p99": "cv_limitado",
    "CV_robusto_MAD": "cv_robusto"
}

# =====================================================
# 11. ANALISIS EN MUESTRA COMPLETA
# =====================================================

resultados_modelos = []
resultados_efectos = []

for nombre, variable in indicadores.items():

    modelo = ajustar_modelo(
        df,
        variable,
        "MUESTRA COMPLETA - " + nombre
    )

    resultados_modelos.append({
        "muestra": "Completa",
        "indicador": nombre,
        "registros": int(modelo.nobs),
        "pseudo_r2": modelo.prsquared,
        "AIC": modelo.aic,
        "BIC": modelo.bic,
        "coef_variabilidad": modelo.params["variabilidad_z"],
        "p_variabilidad": modelo.pvalues["variabilidad_z"],
        "coef_interaccion": modelo.params["interaccion_var"],
        "p_interaccion": modelo.pvalues["interaccion_var"]
    })

    resultados_efectos.extend(
        calcular_efectos(
            modelo,
            df,
            "Completa - " + nombre
        )
    )

# =====================================================
# 12. ANALISIS EN ESTUDIANTES CON MAYOR ACTIVIDAD
# =====================================================

df_alta = df[
    df["media_actividad"] >= 54
].copy()

print("\nMUESTRA DE MAYOR ACTIVIDAD")
print("============================")
print("Umbral de media: 54 clics")
print("Registros:", len(df_alta))

for nombre, variable in indicadores.items():

    modelo = ajustar_modelo(
        df_alta,
        variable,
        "ALTA ACTIVIDAD - " + nombre
    )

    resultados_modelos.append({
        "muestra": "Alta actividad",
        "indicador": nombre,
        "registros": int(modelo.nobs),
        "pseudo_r2": modelo.prsquared,
        "AIC": modelo.aic,
        "BIC": modelo.bic,
        "coef_variabilidad": modelo.params["variabilidad_z"],
        "p_variabilidad": modelo.pvalues["variabilidad_z"],
        "coef_interaccion": modelo.params["interaccion_var"],
        "p_interaccion": modelo.pvalues["interaccion_var"]
    })

    resultados_efectos.extend(
        calcular_efectos(
            modelo,
            df_alta,
            "Alta actividad - " + nombre
        )
    )

# =====================================================
# 13. GUARDAR RESULTADOS
# =====================================================

tabla_modelos = pd.DataFrame(resultados_modelos)

tabla_efectos = pd.DataFrame(resultados_efectos)

tabla_modelos.to_csv(
    SALIDA / "128_comparacion_indicadores.csv",
    index=False
)

tabla_efectos.to_csv(
    SALIDA / "128_efectos_condicionales.csv",
    index=False
)

print("\nCOMPARACION GENERAL")
print("============================")
print(tabla_modelos.round(6).to_string(index=False))

print("\nEFECTOS CONDICIONALES")
print("============================")
print(tabla_efectos.round(6).to_string(index=False))

print("\nARCHIVOS GUARDADOS")
print("============================")
print("128_comparacion_indicadores.csv")
print("128_efectos_condicionales.csv")

print("\nANALISIS FINALIZADO")
