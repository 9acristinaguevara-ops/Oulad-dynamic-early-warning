import numpy as np
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

ARCHIVO_SALIDA = SALIDA / "146_intervalos_confianza_bootstrap_estudiante.csv"

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

REPLICAS = 1000
SEMILLA = 146

# =====================================================
# 2. RECONSTRUIR LAS COHORTES DEL EXPERIMENTO 144
# =====================================================

df = pd.read_parquet(ARCHIVO)

necesarias = VARIABLES + [OBJETIVO, ID, PRESENTACION, "code_module"]
faltantes = [c for c in necesarias if c not in df.columns]

if faltantes:
    raise ValueError("Faltan columnas: " + ", ".join(faltantes))

p = {
    nombre: df[df[PRESENTACION] == nombre].copy()
    for nombre in ["2013B", "2013J", "2014B", "2014J"]
}

if any(x.empty for x in p.values()):
    raise ValueError("Falta uno de los periodos requeridos.")

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

ids = {n: set(x[ID]) for n, x in cohortes.items()}
nombres = list(ids)

for i in range(len(nombres)):
    for j in range(i + 1, len(nombres)):
        if ids[nombres[i]].intersection(ids[nombres[j]]):
            raise AssertionError("Se detecto solapamiento entre cohortes.")

# Cada estudiante aporta un peso total de uno en su cohorte.
def pesos_estudiante(datos):
    return 1.0 / datos.groupby(ID)[ID].transform("size")

pesos = {
    nombre: pesos_estudiante(datos)
    for nombre, datos in cohortes.items()
}

# =====================================================
# 3. ENTRENAR LOS DOS MODELOS Y SUS CALIBRADORES
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

x_cal = calibracion[VARIABLES].fillna(0)
y_cal = calibracion[OBJETIVO].astype(int)

prob_cal_original = modelo_original.predict_proba(x_cal)[:, 1]
prob_cal_ponderada = modelo_ponderado.predict_proba(x_cal)[:, 1]

calibrador_original = LogisticRegression(max_iter=1000)
calibrador_original.fit(
    prob_cal_original.reshape(-1, 1),
    y_cal
)

calibrador_ponderado = LogisticRegression(max_iter=1000)
calibrador_ponderado.fit(
    prob_cal_ponderada.reshape(-1, 1),
    y_cal,
    sample_weight=pesos["calibracion"]
)

# Ambos pipelines usan el umbral 0.15 obtenido en el experimento 145.
# La selección original y ponderada produjo ese mismo umbral.
UMBRAL = 0.15

# =====================================================
# 4. FUNCIONES PARA BOOTSTRAP POR ESTUDIANTE
# =====================================================

def metricas(y, prob, umbral, sample_weight=None):
    pred = (prob >= umbral).astype(int)

    return {
        "AUC": roc_auc_score(y, prob, sample_weight=sample_weight),
        "F1": f1_score(
            y, pred, sample_weight=sample_weight, zero_division=0
        ),
        "precision": precision_score(
            y, pred, sample_weight=sample_weight, zero_division=0
        ),
        "recall": recall_score(
            y, pred, sample_weight=sample_weight, zero_division=0
        ),
        "Brier": brier_score_loss(
            y, prob, sample_weight=sample_weight
        )
    }


def bootstrap_estudiantes(datos, prob, nombre_periodo, nombre_modelo):
    # Guardar filas de cada estudiante. Cada extracción del bootstrap
    # conserva juntas todas las matrículas de esa persona.
    grupos = {
        estudiante: indices
        for estudiante, indices in datos.groupby(ID).indices.items()
    }

    estudiantes = np.array(list(grupos.keys()), dtype=object)
    rng = np.random.default_rng(SEMILLA)

    y_completo = datos[OBJETIVO].astype(int).to_numpy()
    prob_completa = np.asarray(prob)

    resultados = {
        "por_matricula": [],
        "igual_peso_por_estudiante": []
    }

    for replica in range(REPLICAS):
        seleccion = rng.choice(
            estudiantes,
            size=len(estudiantes),
            replace=True
        )

        indices_boot = []
        pesos_boot = []

        # Cada aparición del estudiante en la muestra bootstrap es
        # una unidad independiente del remuestreo.
        for numero, estudiante in enumerate(seleccion):
            indices = grupos[estudiante]
            indices_boot.extend(indices)
            pesos_boot.extend([1.0 / len(indices)] * len(indices))

        indices_boot = np.asarray(indices_boot, dtype=int)
        pesos_boot = np.asarray(pesos_boot, dtype=float)

        y = y_completo[indices_boot]
        prob_rep = prob_completa[indices_boot]

        if len(np.unique(y)) < 2:
            continue

        resultados["por_matricula"].append(
            metricas(y, prob_rep, UMBRAL)
        )

        resultados["igual_peso_por_estudiante"].append(
            metricas(y, prob_rep, UMBRAL, pesos_boot)
        )

    filas = []

    for tipo, lista in resultados.items():
        if len(lista) < REPLICAS * 0.95:
            raise RuntimeError(
                "Demasiadas replicas sin ambas clases en " + nombre_periodo
            )

        tabla = pd.DataFrame(lista)

        for variable in ["AUC", "F1", "precision", "recall", "Brier"]:
            filas.append({
                "modelo": nombre_modelo,
                "periodo": nombre_periodo,
                "ponderacion": tipo,
                "metrica": variable,
                "replicas_validas": len(tabla),
                "estimacion_bootstrap": tabla[variable].mean(),
                "IC95_inferior": tabla[variable].quantile(0.025),
                "IC95_superior": tabla[variable].quantile(0.975)
            })

    return filas


# =====================================================
# 5. PREDICCIONES Y BOOTSTRAP
# =====================================================

resultados = []

pipelines = [
    ("original_por_matricula", modelo_original, calibrador_original),
    ("ponderado_por_estudiante", modelo_ponderado, calibrador_ponderado)
]

for nombre_modelo, modelo, calibrador in pipelines:
    for nombre_periodo, datos in [
        ("prueba_2014B", prueba_b),
        ("prueba_2014J", prueba_j)
    ]:
        x = datos[VARIABLES].fillna(0)
        prob_base = modelo.predict_proba(x)[:, 1]
        prob = calibrador.predict_proba(
            prob_base.reshape(-1, 1)
        )[:, 1]

        print(
            "Bootstrap:",
            nombre_modelo,
            nombre_periodo,
            "| estudiantes:",
            datos[ID].nunique(),
            "| replicas:",
            REPLICAS
        )

        resultados.extend(
            bootstrap_estudiantes(
                datos,
                prob,
                nombre_periodo,
                nombre_modelo
            )
        )

tabla = pd.DataFrame(resultados)
tabla.to_csv(ARCHIVO_SALIDA, index=False)

print("\nINTERVALOS DE CONFIANZA DEL 95%")
print(tabla.round(4).to_string(index=False))
print("\nARCHIVO GUARDADO:")
print(ARCHIVO_SALIDA)
