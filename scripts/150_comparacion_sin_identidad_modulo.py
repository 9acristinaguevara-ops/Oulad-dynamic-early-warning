import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, brier_score_loss
)

# =====================================================
# 1. CONFIGURACION
# =====================================================

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

OBJETIVO = "target_future_withdrawal"
ID = "id_student"
PRESENTACION = "code_presentation"

VARIABLES_ESTUDIANTE = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4",
    "evaluaciones_entregadas_34"
]
VARIABLES_CONTEXTO = ["evaluaciones_programadas_34"]

# Se fijan capacidades operativas para evitar que el modelo
# emita alertas para casi todas las matriculas.
TASAS_OBJETIVO_ALERTA = [0.20, 0.30, 0.40]

# =====================================================
# 2. CARGAR DATOS Y RECONSTRUIR COHORTES ESTRICTAS
# =====================================================

df = pd.read_parquet(ARCHIVO)

necesarias = (
    VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
    + [OBJETIVO, ID, PRESENTACION]
)
faltantes = [c for c in necesarias if c not in df.columns]
if faltantes:
    raise ValueError("Faltan columnas: " + ", ".join(faltantes))

periodos = {
    nombre: df[df[PRESENTACION] == nombre].copy()
    for nombre in ["2013B", "2013J", "2014B", "2014J"]
}
if any(datos.empty for datos in periodos.values()):
    raise ValueError("Falta uno de los periodos requeridos.")

p13b = periodos["2013B"]
p13j = periodos["2013J"]
p14b = periodos["2014B"]
p14j = periodos["2014J"]

ids_13b = set(p13b[ID])
ids_13j = set(p13j[ID])
ids_14b = set(p14b[ID])

divisor = StratifiedGroupKFold(
    n_splits=2, shuffle=True, random_state=42
)
i_cal, i_umbral = next(
    divisor.split(p13j, p13j[OBJETIVO], groups=p13j[ID])
)

calibracion = p13j.iloc[i_cal].copy()
seleccion_umbral = p13j.iloc[i_umbral].copy()
entrenamiento = p13b[~p13b[ID].isin(ids_13j)].copy()
prueba_b = p14b[
    ~p14b[ID].isin(ids_13b.union(ids_13j))
].copy()
prueba_j = p14j[
    ~p14j[ID].isin(ids_13b.union(ids_13j).union(ids_14b))
].copy()

cohortes = {
    "entrenamiento": entrenamiento,
    "calibracion": calibracion,
    "seleccion_umbral": seleccion_umbral,
    "prueba_2014B": prueba_b,
    "prueba_2014J": prueba_j
}

ids = {nombre: set(datos[ID]) for nombre, datos in cohortes.items()}
nombres = list(ids)
for i in range(len(nombres)):
    for j in range(i + 1, len(nombres)):
        if ids[nombres[i]].intersection(ids[nombres[j]]):
            raise AssertionError(
                "Estudiantes compartidos entre "
                + nombres[i] + " y " + nombres[j]
            )

for nombre, datos in cohortes.items():
    if datos.empty or datos[OBJETIVO].nunique() != 2:
        raise ValueError("Cohorte vacia o sin ambas clases: " + nombre)

# =====================================================
# 3. PREPARAR VARIABLES
# =====================================================

def preparar(datos, variables):
    x = datos[variables].copy()
    for columna in variables:
        x[columna] = pd.to_numeric(
            x[columna], errors="coerce"
        ).fillna(0)
    return x

especificaciones = {
    "estudiante_solo": VARIABLES_ESTUDIANTE,
    "contexto_curso_solo": VARIABLES_CONTEXTO,
    "combinado_sin_identidad_modulo":
        VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
}

# =====================================================
# 4. FUNCIONES DE CALIBRACION Y CAPACIDAD DE ALERTA
# =====================================================

def elegir_umbral_por_capacidad(probabilidades, tasa_objetivo):
    # Se prueban los valores de probabilidad observados en seleccion.
    # Se elige el umbral cuya proporcion de alertas quede mas cerca
    # de la capacidad operativa definida.
    candidatos = np.unique(probabilidades)
    mejor_umbral = candidatos[0]
    mejor_diferencia = float("inf")
    tasa_obtenida = 1.0

    for umbral in candidatos:
        tasa = (probabilidades >= umbral).mean()
        diferencia = abs(tasa - tasa_objetivo)

        if diferencia < mejor_diferencia:
            mejor_diferencia = diferencia
            mejor_umbral = umbral
            tasa_obtenida = tasa

    return float(mejor_umbral), float(tasa_obtenida)

def calcular_metricas(nombre, periodo, datos, prob, umbral, tasa_objetivo):
    y = datos[OBJETIVO].astype(int).to_numpy()
    pred = (prob >= umbral).astype(int)

    return {
        "modelo": nombre,
        "periodo": periodo,
        "capacidad_objetivo_alertas": tasa_objetivo,
        "umbral": umbral,
        "tasa_alertas_seleccion": np.nan,
        "matriculas": len(datos),
        "estudiantes_unicos": datos[ID].nunique(),
        "retiros": int(y.sum()),
        "AUC": roc_auc_score(y, prob),
        "F1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "Brier": brier_score_loss(y, prob),
        "alertas": int(pred.sum()),
        "porcentaje_alertas": pred.mean() * 100
    }

# =====================================================
# 5. ENTRENAR, CALIBRAR Y EVALUAR
# =====================================================

y_train = entrenamiento[OBJETIVO].astype(int)
y_cal = calibracion[OBJETIVO].astype(int)
y_sel = seleccion_umbral[OBJETIVO].astype(int)

filas_resultados = []
filas_umbral = []

for nombre, variables in especificaciones.items():
    x_train = preparar(entrenamiento, variables)
    x_cal = preparar(calibracion, variables).reindex(
        columns=x_train.columns, fill_value=0
    )
    x_sel = preparar(seleccion_umbral, variables).reindex(
        columns=x_train.columns, fill_value=0
    )

    modelo = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    modelo.fit(x_train, y_train)

    prob_cal_base = modelo.predict_proba(x_cal)[:, 1]
    calibrador = LogisticRegression(max_iter=1000)
    calibrador.fit(
        prob_cal_base.reshape(-1, 1), y_cal
    )

    prob_sel_base = modelo.predict_proba(x_sel)[:, 1]
    prob_sel = calibrador.predict_proba(
        prob_sel_base.reshape(-1, 1)
    )[:, 1]

    for tasa_objetivo in TASAS_OBJETIVO_ALERTA:
        umbral, tasa_real_seleccion = elegir_umbral_por_capacidad(
            prob_sel, tasa_objetivo
        )

        filas_umbral.append({
            "modelo": nombre,
            "capacidad_objetivo_alertas": tasa_objetivo,
            "umbral": umbral,
            "tasa_alertas_en_seleccion": tasa_real_seleccion * 100,
            "F1_en_seleccion": f1_score(
                y_sel, (prob_sel >= umbral).astype(int),
                zero_division=0
            )
        })

        for periodo, datos in [
            ("2014B", prueba_b), ("2014J", prueba_j)
        ]:
            x_test = preparar(datos, variables).reindex(
                columns=x_train.columns, fill_value=0
            )
            prob_base = modelo.predict_proba(x_test)[:, 1]
            prob_test = calibrador.predict_proba(
                prob_base.reshape(-1, 1)
            )[:, 1]

            fila = calcular_metricas(
                nombre, periodo, datos, prob_test,
                umbral, tasa_objetivo
            )
            fila["tasa_alertas_seleccion"] = (
                tasa_real_seleccion * 100
            )
            filas_resultados.append(fila)

tabla_resultados = pd.DataFrame(filas_resultados)
tabla_umbrales = pd.DataFrame(filas_umbral)

tabla_resultados.to_csv(
    SALIDA / "150_comparacion_capacidad_alertas.csv",
    index=False
)
tabla_umbrales.to_csv(
    SALIDA / "150_umbrales_por_capacidad.csv",
    index=False
)

# =====================================================
# 6. INFORME
# =====================================================

print("\nEXPERIMENTO 150: MODELOS SIN IDENTIDAD DEL MODULO")
print("\nCOHORTES ESTRICTAMENTE INDEPENDIENTES")
for nombre, datos in cohortes.items():
    print(
        nombre,
        "| matriculas:", len(datos),
        "| estudiantes:", datos[ID].nunique(),
        "| retiros:", int(datos[OBJETIVO].sum())
    )

print("\nUMBRALES SEGUN CAPACIDAD DE ALERTAS EN SELECCION")
print(tabla_umbrales.round(4).to_string(index=False))

print("\nRESULTADOS EN ESTUDIANTES NUEVOS")
print(tabla_resultados.round(4).to_string(index=False))

print("\nARCHIVOS GUARDADOS:")
print(SALIDA / "150_comparacion_capacidad_alertas.csv")
print(SALIDA / "150_umbrales_por_capacidad.csv")
