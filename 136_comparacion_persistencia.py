
# 136_comparacion_persistencia.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm
from itertools import combinations
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

    columna = f"clics_semana_{semana}"

    temporal = df[
        [
            "code_module",
            "code_presentation",
            "id_student",
            columna
        ]
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

# Las semanas sin registros permanecen como faltantes.
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
# 5. CREAR TRANSICIONES SEMANALES
# =====================================================

claves = [
    "code_module",
    "code_presentation",
    "id_student"
]

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

print("\nDATOS PARA EL MODELO")
print("Observaciones:", len(pares))
print("Estudiantes:", pares["id_student"].nunique())
print("Grupos:", pares["grupo"].nunique())

# =====================================================
# 6. ESTIMAR MODELO
# =====================================================

modelo = smf.ols(
    """
    actividad_siguiente ~
    actividad_actual * C(semana)
    + C(grupo)
    """,
    data=pares
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": pares["id_student"]
    }
)

# =====================================================
# 7. CONSTRUIR CONTRASTES DE PERSISTENCIA
# =====================================================

parametros = modelo.params
covarianza = modelo.cov_params()

contrastes = {}

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

    contrastes[semana] = contraste

# =====================================================
# 8. PRUEBA CONJUNTA DE IGUALDAD
# =====================================================

# H0: las cuatro persistencias son iguales.
# Equivale a que las tres interacciones sean cero.

nombres_interaccion = [
    f"actividad_actual:C(semana)[T.{semana}]"
    for semana in range(1, 4)
]

matriz_R = np.zeros(
    (3, len(parametros))
)

for fila, nombre in enumerate(nombres_interaccion):

    posicion = parametros.index.get_loc(nombre)

    matriz_R[fila, posicion] = 1.0

prueba_conjunta = modelo.wald_test(
    matriz_R,
    use_f=False,
    scalar=True
)

print("\nPRUEBA CONJUNTA DE IGUALDAD")
print("Hipótesis nula: las cuatro persistencias son iguales")
print("Estadístico chi-cuadrado:", float(prueba_conjunta.statistic))
print("Grados de libertad:", 3)
print("Valor p:", float(prueba_conjunta.pvalue))

# =====================================================
# 9. COMPARACIONES POR PARES
# =====================================================

resultados = []

for semana_a, semana_b in combinations(range(4), 2):

    # Diferencia entre los coeficientes.

    contraste = (
        contrastes[semana_a]
        - contrastes[semana_b]
    )

    diferencia = float(
        contraste @ parametros
    )

    # Error estándar de la diferencia.

    varianza = float(
        contraste @ covarianza @ contraste
    )

    error = np.sqrt(max(varianza, 0))

    z = diferencia / error

    p_valor = 2 * norm.sf(abs(z))

    inferior = diferencia - 1.96 * error
    superior = diferencia + 1.96 * error

    resultados.append({
        "comparacion": f"{semana_a} a {semana_a + 1} vs "
                       f"{semana_b} a {semana_b + 1}",
        "diferencia": diferencia,
        "error_estandar": error,
        "z": z,
        "p_sin_ajustar": p_valor,
        "ic95_inferior": inferior,
        "ic95_superior": superior
    })

tabla = pd.DataFrame(resultados)

# =====================================================
# 10. CORRECCION DE HOLM
# =====================================================

# Ordenar los valores p de menor a mayor.

orden = np.argsort(
    tabla["p_sin_ajustar"].values
)

p_ordenados = (
    tabla["p_sin_ajustar"].values[orden]
)

cantidad = len(p_ordenados)

p_ajustados = np.zeros(cantidad)

maximo_anterior = 0

for posicion in range(cantidad):

    valor = (
        (cantidad - posicion)
        * p_ordenados[posicion]
    )

    valor = min(valor, 1.0)

    valor = max(
        valor,
        maximo_anterior
    )

    p_ajustados[posicion] = valor

    maximo_anterior = valor

# Volver al orden original.

p_holm = np.zeros(cantidad)

p_holm[orden] = p_ajustados

tabla["p_holm"] = p_holm

tabla["significativo_holm"] = (
    tabla["p_holm"] < 0.05
)

# =====================================================
# 11. MOSTRAR RESULTADOS
# =====================================================

print("\nCOMPARACIONES ENTRE TRANSICIONES")

print(
    tabla.to_string(index=False)
)

# =====================================================
# 12. GUARDAR RESULTADOS
# =====================================================

archivo_salida = (
    CARPETA_SALIDA /
    "136_comparacion_persistencia.csv"
)

tabla.to_csv(
    archivo_salida,
    index=False
)

print("\nRESULTADOS GUARDADOS EN:")
print(archivo_salida)

print("\nEXPERIMENTO 136 FINALIZADO")
