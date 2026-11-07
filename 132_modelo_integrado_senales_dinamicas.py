
# 132_modelo_integrado_senales_dinamicas.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from pathlib import Path
from scipy.stats import chi2

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

columnas = [
    "clics_semana_0",
    "clics_semana_1",
    "clics_semana_2",
    "clics_semana_3",
    "clics_semana_4"
]

# Seleccionar trayectorias completas
df = df[
    df[columnas].notna().all(axis=1)
].copy()

print("DATOS")
print("============================")
print("Trayectorias completas:", len(df))

# =====================================================
# 3. CALCULAR INDICADORES INDIVIDUALES
# =====================================================

df["media_clicks"] = df[columnas].mean(axis=1)

df["desviacion_clicks"] = df[columnas].std(axis=1)

df["cv"] = (
    df["desviacion_clicks"] /
    df["media_clicks"]
)

# Transformación logarítmica del promedio
df["log_media"] = np.log1p(df["media_clicks"])

# =====================================================
# 4. CALCULAR PERSISTENCIA Y RECUPERACION
# =====================================================

def calcular_dinamica(fila):

    clics = [
        fila["clics_semana_0"],
        fila["clics_semana_1"],
        fila["clics_semana_2"],
        fila["clics_semana_3"],
        fila["clics_semana_4"]
    ]

    cambios = []

    for i in range(1, len(clics)):

        diferencia = clics[i] - clics[i - 1]

        if diferencia < 0:
            cambios.append(-1)

        elif diferencia > 0:
            cambios.append(1)

        else:
            cambios.append(0)

    persistencia = 0
    recuperaciones = 0

    for i in range(len(cambios) - 1):

        if cambios[i] == -1:

            if cambios[i + 1] == -1:
                persistencia += 1

            elif cambios[i + 1] == 1:
                recuperaciones += 1

    return pd.Series({
        "persistencia": persistencia,
        "recuperaciones": recuperaciones
    })


dinamica = df.apply(
    calcular_dinamica,
    axis=1
)

df = pd.concat(
    [df, dinamica],
    axis=1
)

# =====================================================
# 5. PREPARAR VARIABLES
# =====================================================

# Grupo módulo-presentación
df["grupo"] = (
    df["code_module"].astype(str)
    + "_"
    + df["code_presentation"].astype(str)
)

# Estandarizar variables continuas
variables = [
    "log_media",
    "cv",
    "persistencia",
    "recuperaciones"
]

for variable in variables:

    desviacion = df[variable].std()

    df[variable + "_z"] = (
        df[variable] - df[variable].mean()
    ) / desviacion

# Término cuadrático para controlar
# una posible relación no lineal con el nivel medio
df["log_media_z2"] = df["log_media_z"] ** 2

print("\nDISTRIBUCION DE RESULTADOS")
print("============================")

print(
    df["target_future_withdrawal"]
    .value_counts()
    .sort_index()
)

print("\nGrupos módulo-presentación:", df["grupo"].nunique())

# =====================================================
# 6. DEFINIR LOS TRES MODELOS
# =====================================================

formula1 = (
    "target_future_withdrawal ~ "
    "log_media_z + log_media_z2 + C(grupo)"
)

formula2 = (
    formula1
    + " + cv_z"
)

formula3 = (
    formula2
    + " + persistencia_z + recuperaciones_z"
)

# =====================================================
# 7. ESTIMAR MODELOS
# =====================================================

print("\nESTIMANDO MODELOS")
print("============================")

modelo1 = smf.logit(
    formula1,
    data=df
).fit(
    disp=False,
    maxiter=200,
    cov_type="cluster",
    cov_kwds={"groups": df["grupo"]}
)

print("Modelo 1 estimado")

modelo2 = smf.logit(
    formula2,
    data=df
).fit(
    disp=False,
    maxiter=200,
    cov_type="cluster",
    cov_kwds={"groups": df["grupo"]}
)

print("Modelo 2 estimado")

modelo3 = smf.logit(
    formula3,
    data=df
).fit(
    disp=False,
    maxiter=200,
    cov_type="cluster",
    cov_kwds={"groups": df["grupo"]}
)

print("Modelo 3 estimado")

# =====================================================
# 8. COMPARAR MODELOS
# =====================================================

modelos = [
    ("Modelo 1: nivel medio", modelo1),
    ("Modelo 2: + CV", modelo2),
    ("Modelo 3: + dinamica", modelo3)
]

comparacion = []

for nombre, modelo in modelos:

    pseudo_r2 = 1 - (
        modelo.llf / modelo.llnull
    )

    comparacion.append({

        "modelo": nombre,
        "observaciones": int(modelo.nobs),
        "log_likelihood": modelo.llf,
        "pseudo_R2": pseudo_r2,
        "AIC": modelo.aic,
        "BIC": modelo.bic

    })

comparacion = pd.DataFrame(comparacion)

print("\nCOMPARACION DE MODELOS")
print("============================")

print(
    comparacion.round(5).to_string(index=False)
)

# =====================================================
# 9. PRUEBAS DE RAZON DE VEROSIMILITUD
# =====================================================

def prueba_lr(modelo_reducido, modelo_completo):

    estadistico = 2 * (
        modelo_completo.llf -
        modelo_reducido.llf
    )

    grados_libertad = (
        len(modelo_completo.params)
        - len(modelo_reducido.params)
    )

    valor_p = chi2.sf(
        estadistico,
        grados_libertad
    )

    return estadistico, grados_libertad, valor_p


print("\nPRUEBAS DE RAZON DE VEROSIMILITUD")
print("============================")

lr_12 = prueba_lr(modelo1, modelo2)
lr_23 = prueba_lr(modelo2, modelo3)

print(
    "Modelo 1 frente a Modelo 2:",
    "LR =", round(lr_12[0], 4),
    "| gl =", lr_12[1],
    "| p =", lr_12[2]
)

print(
    "Modelo 2 frente a Modelo 3:",
    "LR =", round(lr_23[0], 4),
    "| gl =", lr_23[1],
    "| p =", lr_23[2]
)

# =====================================================
# 10. COEFICIENTES DEL MODELO FINAL
# =====================================================

print("\nCOEFICIENTES DEL MODELO 3")
print("============================")

variables_finales = [
    "log_media_z",
    "log_media_z2",
    "cv_z",
    "persistencia_z",
    "recuperaciones_z"
]

resultados = []

for variable in variables_finales:

    coef = modelo3.params[variable]
    error = modelo3.bse[variable]
    valor_p = modelo3.pvalues[variable]

    inferior = coef - 1.96 * error
    superior = coef + 1.96 * error

    resultados.append({

        "variable": variable,
        "coeficiente": coef,
        "error_estandar_cluster": error,
        "odds_ratio": np.exp(coef),
        "IC95_inferior_OR": np.exp(inferior),
        "IC95_superior_OR": np.exp(superior),
        "valor_p": valor_p

    })

resultados = pd.DataFrame(resultados)

print(
    resultados.round(5).to_string(index=False)
)

# =====================================================
# 11. GUARDAR RESULTADOS
# =====================================================

comparacion.to_csv(
    SALIDA / "132_comparacion_modelos.csv",
    index=False
)

resultados.to_csv(
    SALIDA / "132_coeficientes_modelo_final.csv",
    index=False
)

df[
    [
        "code_module",
        "code_presentation",
        "id_student",
        "target_future_withdrawal",
        "media_clicks",
        "cv",
        "persistencia",
        "recuperaciones"
    ]
].to_csv(
    SALIDA / "132_datos_modelo_integrado.csv",
    index=False
)

print("\nARCHIVOS GUARDADOS")
print("============================")

print("132_comparacion_modelos.csv")
print("132_coeficientes_modelo_final.csv")
print("132_datos_modelo_integrado.csv")

print("\nANALISIS FINALIZADO")
