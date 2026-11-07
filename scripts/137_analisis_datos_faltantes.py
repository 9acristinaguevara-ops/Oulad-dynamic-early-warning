
# 137_analisis_datos_faltantes.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

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

# =====================================================
# 3. CONSTRUIR FORMATO LONGITUDINAL
# =====================================================

registros = []

for semana in range(5):

    columna = f"actividad_registrada_semana_{semana}"

    temporal = df[
        [
            "code_module",
            "code_presentation",
            "id_student",
            "target_future_withdrawal",
            columna
        ]
    ].copy()

    temporal = temporal.rename(
        columns={
            columna: "actividad_observada"
        }
    )

    temporal["semana"] = semana

    registros.append(temporal)

longitudinal = pd.concat(
    registros,
    ignore_index=True
)

# Convertir el indicador a valores enteros.
# 1 = actividad registrada.
# 0 = actividad no registrada.

longitudinal["actividad_observada"] = (
    longitudinal["actividad_observada"].astype(int)
)

# Identificar cada módulo y presentación.

longitudinal["grupo"] = (
    longitudinal["code_module"].astype(str)
    + "_"
    + longitudinal["code_presentation"].astype(str)
)

print("\nREGISTROS LONGITUDINALES")
print(longitudinal.shape)

# =====================================================
# 4. COBERTURA POR GRUPO Y SEMANA
# =====================================================

cobertura = (
    longitudinal
    .groupby(
        ["target_future_withdrawal", "semana"]
    )["actividad_observada"]
    .agg(
        estudiantes="count",
        registros_observados="sum",
        porcentaje_observado="mean"
    )
    .reset_index()
)

cobertura["porcentaje_observado"] = (
    cobertura["porcentaje_observado"] * 100
)

print("\nCOBERTURA POR GRUPO Y SEMANA")

print(
    cobertura.to_string(index=False)
)

# =====================================================
# 5. DIFERENCIAS DE COBERTURA
# =====================================================

tabla_ancha = cobertura.pivot(
    index="semana",
    columns="target_future_withdrawal",
    values="porcentaje_observado"
)

tabla_ancha = tabla_ancha.rename(
    columns={
        0: "no_abandono",
        1: "abandono_futuro"
    }
)

tabla_ancha["diferencia_puntos_porcentuales"] = (
    tabla_ancha["abandono_futuro"]
    - tabla_ancha["no_abandono"]
)

print("\nDIFERENCIAS DE COBERTURA")

print(
    tabla_ancha.to_string()
)

# =====================================================
# 6. PATRONES DE SEMANAS OBSERVADAS
# =====================================================

columnas_indicadores = [
    f"actividad_registrada_semana_{semana}"
    for semana in range(5)
]

matriz = df[columnas_indicadores].copy()

matriz.columns = [
    f"semana_{semana}"
    for semana in range(5)
]

patrones = (
    matriz.astype(int)
    .astype(str)
    .agg("".join, axis=1)
)

df_patrones = pd.DataFrame({
    "patron": patrones,
    "target": df["target_future_withdrawal"]
})

resumen_patrones = (
    df_patrones
    .groupby(["target", "patron"])
    .size()
    .reset_index(name="estudiantes")
)

print("\nPATRONES DE OBSERVACION")

print(
    resumen_patrones
    .sort_values(
        ["target", "estudiantes"],
        ascending=[True, False]
    )
    .to_string(index=False)
)

# =====================================================
# 7. MODELO LOGISTICO DE DISPONIBILIDAD
# =====================================================

modelo = smf.glm(
    """
    actividad_observada ~
    target_future_withdrawal * C(semana)
    + C(grupo)
    """,
    data=longitudinal,
    family=sm.families.Binomial()
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": longitudinal["id_student"]
    }
)

print("\nMODELO LOGISTICO DE DISPONIBILIDAD")

print(modelo.summary())

# =====================================================
# 8. CALCULAR ODDS RATIOS
# =====================================================

coeficientes = modelo.params
errores = modelo.bse
p_valores = modelo.pvalues

resultados_modelo = pd.DataFrame({
    "variable": coeficientes.index,
    "coeficiente": coeficientes.values,
    "error_estandar": errores.values,
    "odds_ratio": np.exp(coeficientes.values),
    "p_valor": p_valores.values
})

print("\nCOEFICIENTES Y ODDS RATIOS")

print(
    resultados_modelo.to_string(index=False)
)

# =====================================================
# 9. GUARDAR RESULTADOS
# =====================================================

cobertura.to_csv(
    CARPETA_SALIDA / "137_cobertura_semanal.csv",
    index=False
)

tabla_ancha.to_csv(
    CARPETA_SALIDA / "137_diferencias_cobertura.csv"
)

resumen_patrones.to_csv(
    CARPETA_SALIDA / "137_patrones_observacion.csv",
    index=False
)

resultados_modelo.to_csv(
    CARPETA_SALIDA / "137_modelo_disponibilidad.csv",
    index=False
)

print("\nARCHIVOS GUARDADOS EN:")
print(CARPETA_SALIDA)

print("\nEXPERIMENTO 137 FINALIZADO")
