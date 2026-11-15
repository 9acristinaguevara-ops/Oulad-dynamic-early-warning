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
# 1. RUTAS Y CONFIGURACION
# =====================================================

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVO_RESULTADOS = SALIDA / "145_sensibilidad_peso_estudiante.csv"
ARCHIVO_COHORTES = SALIDA / "145_resumen_pesos_cohortes.csv"
ARCHIVO_UMBRALES = SALIDA / "145_comparacion_umbrales.csv"

VARIABLES = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4",
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34"
]

OBJETIVO = "target_future_withdrawal"
ID = "id_student"
PRESENTACION = "code_presentation"

# =====================================================
# 2. CARGAR DATOS Y RECONSTRUIR LAS COHORTES
# =====================================================

df = pd.read_parquet(ARCHIVO)

columnas = VARIABLES + [OBJETIVO, ID, PRESENTACION, "code_module"]
faltantes = [c for c in columnas if c not in df.columns]

if faltantes:
    raise ValueError("Faltan columnas: " + ", ".join(faltantes))

if df[[OBJETIVO, ID, PRESENTACION, "code_module"]].isnull().any().any():
    raise ValueError("Hay valores ausentes en columnas criticas.")

if not set(df[OBJETIVO].unique()).issubset({0, 1}):
    raise ValueError("El objetivo debe contener solamente 0 y 1.")

p = {
    nombre: df[df[PRESENTACION] == nombre].copy()
    for nombre in ["2013B", "2013J", "2014B", "2014J"]
}

if any(x.empty for x in p.values()):
    raise ValueError("Falta uno de los periodos necesarios.")

# Separacion de 2013J por estudiante, igual que en el script 144.
divisor = StratifiedGroupKFold(
    n_splits=2,
    shuffle=True,
    random_state=42
)

i_cal, i_umbral = next(
    divisor.split(
        p["2013J"][VARIABLES],
        p["2013J"][OBJETIVO],
        groups=p["2013J"][ID]
    )
)

calibracion = p["2013J"].iloc[i_cal].copy()
seleccion_umbral = p["2013J"].iloc[i_umbral].copy()

ids_2013B = set(p["2013B"][ID])
ids_2013J = set(p["2013J"][ID])
ids_2014B = set(p["2014B"][ID])

entrenamiento = p["2013B"][
    ~p["2013B"][ID].isin(ids_2013J)
].copy()

prueba_b = p["2014B"][
    ~p["2014B"][ID].isin(ids_2013B.union(ids_2013J))
].copy()

prueba_j = p["2014J"][
    ~p["2014J"][ID].isin(ids_2013B.union(ids_2013J).union(ids_2014B))
].copy()

cohortes = {
    "entrenamiento": entrenamiento,
    "calibracion": calibracion,
    "seleccion_umbral": seleccion_umbral,
    "prueba_2014B": prueba_b,
    "prueba_2014J": prueba_j
}

# Confirmar que no haya estudiantes compartidos entre cohortes.
ids = {n: set(x[ID]) for n, x in cohortes.items()}
nombres = list(ids)

for i in range(len(nombres)):
    for j in range(i + 1, len(nombres)):
        if ids[nombres[i]].intersection(ids[nombres[j]]):
            raise AssertionError(
                "Solapamiento entre " + nombres[i] + " y " + nombres[j]
            )

for nombre, datos in cohortes.items():
    if datos.empty or datos[OBJETIVO].nunique() != 2:
        raise ValueError("Cohorte vacia o sin ambas clases: " + nombre)

# =====================================================
# 3. CALCULAR PESOS: CADA ESTUDIANTE SUMA 1 POR COHORTE
# =====================================================

def pesos_por_estudiante(datos):
    cantidad = datos.groupby(ID)[ID].transform("size")
    return 1.0 / cantidad

pesos = {
    nombre: pesos_por_estudiante(datos)
    for nombre, datos in cohortes.items()
}

filas_pesos = []

for nombre, datos in cohortes.items():
    filas_pesos.append({
        "cohorte": nombre,
        "matriculas": len(datos),
        "estudiantes": datos[ID].nunique(),
        "suma_pesos": pesos[nombre].sum(),
        "peso_minimo": pesos[nombre].min(),
        "peso_maximo": pesos[nombre].max(),
        "clase_1_matriculas": int(datos[OBJETIVO].sum()),
        "clase_1_ponderada": float(
            (datos[OBJETIVO] * pesos[nombre]).sum()
        )
    })

tabla_pesos = pd.DataFrame(filas_pesos)
tabla_pesos.to_csv(ARCHIVO_COHORTES, index=False)

# =====================================================
# 4. FUNCIONES DE CALIBRACION Y SELECCION DE UMBRAL
# =====================================================

def ajustar_calibrador(modelo, datos_cal, usar_pesos):
    x = datos_cal[VARIABLES].fillna(0)
    y = datos_cal[OBJETIVO].astype(int)
    prob = modelo.predict_proba(x)[:, 1]

    calibrador = LogisticRegression(max_iter=1000)

    if usar_pesos:
        calibrador.fit(
            prob.reshape(-1, 1),
            y,
            sample_weight=pesos["calibracion"]
        )
    else:
        calibrador.fit(prob.reshape(-1, 1), y)

    return calibrador


def elegir_umbral(modelo, calibrador, usar_pesos):
    x = seleccion_umbral[VARIABLES].fillna(0)
    y = seleccion_umbral[OBJETIVO].astype(int)

    prob_base = modelo.predict_proba(x)[:, 1]
    prob = calibrador.predict_proba(prob_base.reshape(-1, 1))[:, 1]

    peso = pesos["seleccion_umbral"] if usar_pesos else None

    lista_umbrales = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
    mejor_umbral = lista_umbrales[0]
    mejor_f1 = -1

    for umbral in lista_umbrales:
        pred = (prob >= umbral).astype(int)
        f1 = f1_score(
            y,
            pred,
            sample_weight=peso,
            zero_division=0
        )

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral

    return mejor_umbral, mejor_f1


# =====================================================
# 5. FUNCION DE EVALUACION
# =====================================================

def evaluar(nombre_modelo, periodo, datos, prob, umbral, ponderar_metricas):
    y = datos[OBJETIVO].astype(int)
    pred = (prob >= umbral).astype(int)

    peso = pesos[periodo] if ponderar_metricas else None

    return {
        "modelo": nombre_modelo,
        "periodo": periodo,
        "tipo_metricas": (
            "igual_peso_por_estudiante"
            if ponderar_metricas else "por_matricula"
        ),
        "matriculas": len(datos),
        "estudiantes_unicos": datos[ID].nunique(),
        "retiros_matriculas": int(y.sum()),
        "retiros_ponderados": float((y * pesos[periodo]).sum()),
        "umbral": umbral,
        "AUC": roc_auc_score(y, prob, sample_weight=peso),
        "F1": f1_score(y, pred, sample_weight=peso, zero_division=0),
        "precision": precision_score(
            y, pred, sample_weight=peso, zero_division=0
        ),
        "recall": recall_score(
            y, pred, sample_weight=peso, zero_division=0
        ),
        "Brier": brier_score_loss(y, prob, sample_weight=peso),
        "alertas_matriculas": int(pred.sum()),
        "porcentaje_alertas_matriculas": float(pred.mean() * 100),
        "porcentaje_alertas_ponderado": float(
            (pred * pesos[periodo]).sum()
            / pesos[periodo].sum() * 100
        )
    }


# =====================================================
# 6. ENTRENAR DOS PIPELINES: ORIGINAL Y PONDERADO
# =====================================================

x_train = entrenamiento[VARIABLES].fillna(0)
y_train = entrenamiento[OBJETIVO].astype(int)

modelo_original = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
modelo_original.fit(x_train, y_train)

modelo_ponderado = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
modelo_ponderado.fit(
    x_train,
    y_train,
    sample_weight=pesos["entrenamiento"]
)

calibrador_original = ajustar_calibrador(
    modelo_original, calibracion, False
)
calibrador_ponderado = ajustar_calibrador(
    modelo_ponderado, calibracion, True
)

umbral_original, f1_original = elegir_umbral(
    modelo_original, calibrador_original, False
)
umbral_ponderado, f1_ponderado = elegir_umbral(
    modelo_ponderado, calibrador_ponderado, True
)

tabla_umbrales = pd.DataFrame([
    {
        "pipeline": "original_por_matricula",
        "umbral": umbral_original,
        "F1_seleccion": f1_original
    },
    {
        "pipeline": "ponderado_por_estudiante",
        "umbral": umbral_ponderado,
        "F1_seleccion": f1_ponderado
    }
])
tabla_umbrales.to_csv(ARCHIVO_UMBRALES, index=False)

# =====================================================
# 7. EVALUAR AMBOS PIPELINES CON DOS PONDERACIONES
# =====================================================

resultados = []

pipelines = [
    (
        "original_por_matricula",
        modelo_original,
        calibrador_original,
        umbral_original
    ),
    (
        "ponderado_por_estudiante",
        modelo_ponderado,
        calibrador_ponderado,
        umbral_ponderado
    )
]

for nombre, modelo, calibrador, umbral in pipelines:
    for periodo, datos in [
        ("prueba_2014B", prueba_b),
        ("prueba_2014J", prueba_j)
    ]:
        x = datos[VARIABLES].fillna(0)
        prob_base = modelo.predict_proba(x)[:, 1]
        prob = calibrador.predict_proba(
            prob_base.reshape(-1, 1)
        )[:, 1]

        resultados.append(
            evaluar(
                nombre, periodo, datos, prob, umbral, False
            )
        )
        resultados.append(
            evaluar(
                nombre, periodo, datos, prob, umbral, True
            )
        )

tabla_resultados = pd.DataFrame(resultados)
tabla_resultados.to_csv(ARCHIVO_RESULTADOS, index=False)

# =====================================================
# 8. INFORME
# =====================================================

print("\nSENSIBILIDAD CON IGUAL PESO POR ESTUDIANTE")
print("\nPESOS POR COHORTE")
print(tabla_pesos.round(4).to_string(index=False))

print("\nUMBRALES SELECCIONADOS")
print(tabla_umbrales.round(4).to_string(index=False))

print("\nRESULTADOS COMPARATIVOS")
print(tabla_resultados.round(4).to_string(index=False))

print("\nARCHIVOS GUARDADOS:")
print(ARCHIVO_COHORTES)
print(ARCHIVO_UMBRALES)
print(ARCHIVO_RESULTADOS)
