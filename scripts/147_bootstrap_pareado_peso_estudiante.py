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

ARCHIVO_SALIDA = SALIDA / "147_diferencias_pareadas_bootstrap.csv"

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
SEMILLA = 147
UMBRAL = 0.15

# =====================================================
# 2. RECONSTRUIR LAS COHORTES ESTRICTAS
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
            raise AssertionError("Hay estudiantes compartidos entre cohortes.")

def pesos_estudiante(datos):
    return 1.0 / datos.groupby(ID)[ID].transform("size")

pesos = {
    nombre: pesos_estudiante(datos)
    for nombre, datos in cohortes.items()
}

# =====================================================
# 3. ENTRENAR MODELO POR MATRICULA Y MODELO PONDERADO
# =====================================================

x_train = entrenamiento[VARIABLES].fillna(0)
y_train = entrenamiento[OBJETIVO].astype(int)

modelo_matricula = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
modelo_matricula.fit(x_train, y_train)

modelo_estudiante = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
modelo_estudiante.fit(
    x_train,
    y_train,
    sample_weight=pesos["entrenamiento"]
)

x_cal = calibracion[VARIABLES].fillna(0)
y_cal = calibracion[OBJETIVO].astype(int)

prob_cal_matricula = modelo_matricula.predict_proba(x_cal)[:, 1]
prob_cal_estudiante = modelo_estudiante.predict_proba(x_cal)[:, 1]

calibrador_matricula = LogisticRegression(max_iter=1000)
calibrador_matricula.fit(
    prob_cal_matricula.reshape(-1, 1),
    y_cal
)

calibrador_estudiante = LogisticRegression(max_iter=1000)
calibrador_estudiante.fit(
    prob_cal_estudiante.reshape(-1, 1),
    y_cal,
    sample_weight=pesos["calibracion"]
)

# =====================================================
# 4. METRICAS Y BOOTSTRAP PAREADO POR ESTUDIANTE
# =====================================================

def calcular_metricas(y, prob, sample_weight=None):
    pred = (prob >= UMBRAL).astype(int)

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


def bootstrap_pareado(datos, prob_matricula, prob_estudiante, periodo):
    grupos = {
        estudiante: indices
        for estudiante, indices in datos.groupby(ID).indices.items()
    }

    estudiantes = np.array(list(grupos.keys()), dtype=object)
    rng = np.random.default_rng(SEMILLA)

    y_completo = datos[OBJETIVO].astype(int).to_numpy()
    prob_a = np.asarray(prob_matricula)
    prob_b = np.asarray(prob_estudiante)

    diferencias = {
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

        for estudiante in seleccion:
            indices = grupos[estudiante]
            indices_boot.extend(indices)
            pesos_boot.extend([1.0 / len(indices)] * len(indices))

        indices_boot = np.asarray(indices_boot, dtype=int)
        pesos_boot = np.asarray(pesos_boot, dtype=float)

        y = y_completo[indices_boot]
        if len(np.unique(y)) < 2:
            continue

        a = prob_a[indices_boot]
        b = prob_b[indices_boot]

        metricas_a = calcular_metricas(y, a)
        metricas_b = calcular_metricas(y, b)

        metricas_a_p = calcular_metricas(y, a, pesos_boot)
        metricas_b_p = calcular_metricas(y, b, pesos_boot)

        for tipo, ma, mb in [
            ("por_matricula", metricas_a, metricas_b),
            ("igual_peso_por_estudiante", metricas_a_p, metricas_b_p)
        ]:
            diferencias[tipo].append({
                metrica: mb[metrica] - ma[metrica]
                for metrica in ma
            })

    filas = []

    for tipo, lista in diferencias.items():
        if len(lista) < REPLICAS * 0.95:
            raise RuntimeError("Demasiadas replicas bootstrap invalidas.")

        tabla = pd.DataFrame(lista)

        for metrica in ["AUC", "F1", "precision", "recall", "Brier"]:
            valores = tabla[metrica]
            filas.append({
                "periodo": periodo,
                "ponderacion": tipo,
                "diferencia": "ponderado_estudiante - original_matricula",
                "metrica": metrica,
                "replicas_validas": len(tabla),
                "diferencia_media": valores.mean(),
                "IC95_inferior": valores.quantile(0.025),
                "IC95_superior": valores.quantile(0.975),
                "probabilidad_diferencia_mayor_cero": (valores > 0).mean()
            })

    return filas


# =====================================================
# 5. PREDICCIONES EN LOS DOS PERIODOS DE PRUEBA
# =====================================================

resultados = []

for periodo, datos in [
    ("2014B", prueba_b),
    ("2014J", prueba_j)
]:
    x = datos[VARIABLES].fillna(0)

    base_a = modelo_matricula.predict_proba(x)[:, 1]
    prob_a = calibrador_matricula.predict_proba(
        base_a.reshape(-1, 1)
    )[:, 1]

    base_b = modelo_estudiante.predict_proba(x)[:, 1]
    prob_b = calibrador_estudiante.predict_proba(
        base_b.reshape(-1, 1)
    )[:, 1]

    print(
        "Bootstrap pareado:",
        periodo,
        "| estudiantes:",
        datos[ID].nunique(),
        "| replicas:",
        REPLICAS
    )

    resultados.extend(
        bootstrap_pareado(datos, prob_a, prob_b, periodo)
    )

tabla = pd.DataFrame(resultados)
tabla.to_csv(ARCHIVO_SALIDA, index=False)

print("\nDIFERENCIAS PAREADAS: MODELO PONDERADO MENOS MODELO ORIGINAL")
print(tabla.round(5).to_string(index=False))

print("\nARCHIVO GUARDADO:")
print(ARCHIVO_SALIDA)
