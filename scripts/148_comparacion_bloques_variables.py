import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)

# =====================================================
# 1. CONFIGURACION
# =====================================================

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVO_RESULTADOS = SALIDA / "148_comparacion_bloques_variables.csv"
ARCHIVO_COHORTES = SALIDA / "148_resumen_cohortes.csv"
ARCHIVO_UMBRALES = SALIDA / "148_umbrales_por_modelo.csv"

OBJETIVO = "target_future_withdrawal"
ID = "id_student"
PRESENTACION = "code_presentation"
MODULO = "code_module"

# Variables individuales de actividad y participación del estudiante.
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

# Variables que describen el contexto del curso.
# evaluaciones_programadas_34 es constante dentro de cada modulo-presentacion.
VARIABLES_CONTEXTO = ["evaluaciones_programadas_34"]

UMBRALES = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

# =====================================================
# 2. CARGAR Y VALIDAR DATOS
# =====================================================

df = pd.read_parquet(ARCHIVO)

necesarias = (
    VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
    + [OBJETIVO, ID, PRESENTACION, MODULO]
)
faltantes = [c for c in necesarias if c not in df.columns]

if faltantes:
    raise ValueError("Faltan columnas: " + ", ".join(faltantes))

if df[[OBJETIVO, ID, PRESENTACION, MODULO]].isnull().any().any():
    raise ValueError("Hay valores ausentes en el objetivo o las claves.")

if not set(df[OBJETIVO].unique()).issubset({0, 1}):
    raise ValueError("El objetivo debe contener solamente 0 y 1.")

periodos = {
    nombre: df[df[PRESENTACION] == nombre].copy()
    for nombre in ["2013B", "2013J", "2014B", "2014J"]
}
if any(x.empty for x in periodos.values()):
    raise ValueError("Falta uno de los periodos requeridos.")

# =====================================================
# 3. RECONSTRUIR LAS MISMAS COHORTES ESTRICTAS DEL SCRIPT 144
# =====================================================

p13b = periodos["2013B"]
p13j = periodos["2013J"]
p14b = periodos["2014B"]
p14j = periodos["2014J"]

ids_13b = set(p13b[ID])
ids_13j = set(p13j[ID])
ids_14b = set(p14b[ID])

divisor = StratifiedGroupKFold(
    n_splits=2,
    shuffle=True,
    random_state=42
)
i_cal, i_umbral = next(
    divisor.split(p13j, p13j[OBJETIVO], groups=p13j[ID])
)

calibracion = p13j.iloc[i_cal].copy()
seleccion_umbral = p13j.iloc[i_umbral].copy()

entrenamiento = p13b[~p13b[ID].isin(ids_13j)].copy()
prueba_b = p14b[~p14b[ID].isin(ids_13b.union(ids_13j))].copy()
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

# Verificar independencia de estudiantes entre las cinco cohortes.
ids = {nombre: set(datos[ID]) for nombre, datos in cohortes.items()}
nombres = list(ids)
for i in range(len(nombres)):
    for j in range(i + 1, len(nombres)):
        if ids[nombres[i]].intersection(ids[nombres[j]]):
            raise AssertionError(
                "Hay estudiantes compartidos entre "
                + nombres[i] + " y " + nombres[j]
            )

for nombre, datos in cohortes.items():
    if datos.empty or datos[OBJETIVO].nunique() != 2:
        raise ValueError("Cohorte vacia o sin ambas clases: " + nombre)

filas_cohortes = []
for nombre, datos in cohortes.items():
    filas_cohortes.append({
        "cohorte": nombre,
        "matriculas": len(datos),
        "estudiantes_unicos": datos[ID].nunique(),
        "retiros": int(datos[OBJETIVO].sum()),
        "tasa_retiro": datos[OBJETIVO].mean()
    })

tabla_cohortes = pd.DataFrame(filas_cohortes)
tabla_cohortes.to_csv(ARCHIVO_COHORTES, index=False)

# =====================================================
# 4. PREPARAR BLOQUES DE VARIABLES SIN MIRAR LOS RESULTADOS
# =====================================================

# Las categorias de modulo se aprenden solo con entrenamiento.
modulos_entrenamiento = sorted(entrenamiento[MODULO].astype(str).unique())

def preparar_variables(datos, variables_numericas, incluir_modulo):
    x = datos[variables_numericas].copy()

    for columna in variables_numericas:
        x[columna] = pd.to_numeric(x[columna], errors="coerce").fillna(0)

    if incluir_modulo:
        modulo = pd.Categorical(
            datos[MODULO].astype(str),
            categories=modulos_entrenamiento
        )
        dummies = pd.get_dummies(
            modulo,
            prefix="modulo",
            dtype=int
        )
        dummies.index = datos.index
        x = pd.concat([x, dummies], axis=1)

    return x

# Modelo individual: señales de actividad/participacion.
# Modelo contextual: evaluaciones programadas + identidad del modulo.
# Modelo combinado: ambos bloques.
especificaciones = {
    "estudiante_solo": {
        "numericas": VARIABLES_ESTUDIANTE,
        "modulo": False
    },
    "contexto_curso_solo": {
        "numericas": VARIABLES_CONTEXTO,
        "modulo": True
    },
    "combinado": {
        "numericas": VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO,
        "modulo": True
    }
}

# =====================================================
# 5. FUNCIONES DE AJUSTE, CALIBRACION Y UMBRAL
# =====================================================

def ajustar_modelo(x, y):
    modelo = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    modelo.fit(x, y)
    return modelo

def ajustar_calibrador(probabilidades, y):
    calibrador = LogisticRegression(max_iter=1000)
    calibrador.fit(probabilidades.reshape(-1, 1), y)
    return calibrador

def seleccionar_umbral(probabilidades, y):
    mejor_umbral = UMBRALES[0]
    mejor_f1 = -1

    for umbral in UMBRALES:
        pred = (probabilidades >= umbral).astype(int)
        valor_f1 = f1_score(y, pred, zero_division=0)

        if valor_f1 > mejor_f1:
            mejor_f1 = valor_f1
            mejor_umbral = umbral

    return mejor_umbral, mejor_f1

def calcular_resultado(nombre, periodo, datos, prob, umbral):
    y = datos[OBJETIVO].astype(int)
    pred = (prob >= umbral).astype(int)

    return {
        "modelo": nombre,
        "periodo": periodo,
        "matriculas": len(datos),
        "estudiantes_unicos": datos[ID].nunique(),
        "retiros": int(y.sum()),
        "umbral": umbral,
        "AUC": roc_auc_score(y, prob),
        "F1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "Brier": brier_score_loss(y, prob),
        "alertas": int(pred.sum()),
        "porcentaje_alertas": pred.mean() * 100
    }

# =====================================================
# 6. ENTRENAR Y EVALUAR LOS TRES MODELOS
# =====================================================

resultados = []
umbrales_modelos = []

y_train = entrenamiento[OBJETIVO].astype(int)
y_cal = calibracion[OBJETIVO].astype(int)
y_umbral = seleccion_umbral[OBJETIVO].astype(int)

for nombre, especificacion in especificaciones.items():
    numericas = especificacion["numericas"]
    incluir_modulo = especificacion["modulo"]

    x_train = preparar_variables(entrenamiento, numericas, incluir_modulo)
    x_cal = preparar_variables(calibracion, numericas, incluir_modulo)
    x_umbral = preparar_variables(seleccion_umbral, numericas, incluir_modulo)

    # Asegurar exactamente las mismas columnas en todos los conjuntos.
    columnas = x_train.columns
    x_cal = x_cal.reindex(columns=columnas, fill_value=0)
    x_umbral = x_umbral.reindex(columns=columnas, fill_value=0)

    modelo = ajustar_modelo(x_train, y_train)

    prob_cal_base = modelo.predict_proba(x_cal)[:, 1]
    calibrador = ajustar_calibrador(prob_cal_base, y_cal)

    prob_umbral_base = modelo.predict_proba(x_umbral)[:, 1]
    prob_umbral_cal = calibrador.predict_proba(
        prob_umbral_base.reshape(-1, 1)
    )[:, 1]

    umbral, f1_seleccion = seleccionar_umbral(
        prob_umbral_cal, y_umbral
    )

    umbrales_modelos.append({
        "modelo": nombre,
        "variables_numericas": " ; ".join(numericas),
        "incluye_modulo": incluir_modulo,
        "numero_predictores": len(columnas),
        "umbral": umbral,
        "F1_seleccion": f1_seleccion
    })

    print(
        "\nModelo:", nombre,
        "| predictores:", len(columnas),
        "| umbral:", umbral,
        "| F1 seleccion:", round(f1_seleccion, 4)
    )

    for periodo, datos in [
        ("2014B", prueba_b),
        ("2014J", prueba_j)
    ]:
        x_test = preparar_variables(datos, numericas, incluir_modulo)
        x_test = x_test.reindex(columns=columnas, fill_value=0)

        prob_test_base = modelo.predict_proba(x_test)[:, 1]
        prob_test_cal = calibrador.predict_proba(
            prob_test_base.reshape(-1, 1)
        )[:, 1]

        resultados.append(
            calcular_resultado(
                nombre, periodo, datos, prob_test_cal, umbral
            )
        )

tabla_resultados = pd.DataFrame(resultados)
tabla_umbrales = pd.DataFrame(umbrales_modelos)

tabla_resultados.to_csv(ARCHIVO_RESULTADOS, index=False)
tabla_umbrales.to_csv(ARCHIVO_UMBRALES, index=False)

# =====================================================
# 7. INFORME
# =====================================================

print("\nCOMPARACION DE BLOQUES DE VARIABLES")
print("\nCOHORTES ESTRICTAMENTE INDEPENDIENTES")
print(tabla_cohortes.round(4).to_string(index=False))

print("\nUMBRALES Y PREDICTORES")
print(tabla_umbrales.round(4).to_string(index=False))

print("\nRESULTADOS EN ESTUDIANTES NUEVOS")
print(tabla_resultados.round(4).to_string(index=False))

print("\nARCHIVOS GUARDADOS:")
print(ARCHIVO_COHORTES)
print(ARCHIVO_UMBRALES)
print(ARCHIVO_RESULTADOS)
