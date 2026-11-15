import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.special import expit, logit
from scipy.optimize import brentq, minimize

# =====================================================
# EXPERIMENTO 152: DIAGNOSTICO DE CALIBRACION ESTRICTO
# Reproduce las cohortes y el modelo combinado del experimento 150.
# No modifica los archivos de experimentos anteriores.
# =====================================================

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

OBJETIVO = "target_future_withdrawal"
ID = "id_student"
PRESENTACION = "code_presentation"
REPLICAS = 1000
SEMILLA = 152

VARIABLES = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4",
    "evaluaciones_entregadas_34",
    "evaluaciones_programadas_34"
]

# 1. Reconstruir exactamente las cohortes estrictas del experimento 150.
df = pd.read_parquet(ARCHIVO)

periodos = {
    nombre: df[df[PRESENTACION] == nombre].copy()
    for nombre in ["2013B", "2013J", "2014B", "2014J"]
}
if any(d.empty for d in periodos.values()):
    raise ValueError("Falta un periodo requerido.")

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
ids = {n: set(d[ID]) for n, d in cohortes.items()}
nombres = list(ids)
for i in range(len(nombres)):
    for j in range(i + 1, len(nombres)):
        if ids[nombres[i]].intersection(ids[nombres[j]]):
            raise AssertionError("Hay estudiantes compartidos entre cohortes.")

# 2. Entrenar el mismo modelo combinado del experimento 150.
def preparar(datos):
    x = datos[VARIABLES].copy()
    for c in VARIABLES:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    return x

x_train = preparar(entrenamiento)
y_train = entrenamiento[OBJETIVO].astype(int)

modelo = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
)
modelo.fit(x_train, y_train)

prob_cal_base = modelo.predict_proba(preparar(calibracion))[:, 1]
y_cal = calibracion[OBJETIVO].astype(int).to_numpy()

calibrador = LogisticRegression(max_iter=1000)
calibrador.fit(prob_cal_base.reshape(-1, 1), y_cal)

# 3. Funciones de diagnóstico.
def calibracion_intercepto(y, p):
    # Intercepto con pendiente fijada en 1.
    z = logit(np.clip(p, 1e-10, 1 - 1e-10))
    prevalencia = np.mean(y)

    def diferencia(a):
        return np.mean(expit(z + a)) - prevalencia

    return float(brentq(diferencia, -30, 30))

def calibracion_pendiente(y, p):
    # Ajusta logit(P(Y=1)) = intercepto + pendiente * logit(p).
    z = logit(np.clip(p, 1e-10, 1 - 1e-10))
    x = np.column_stack([np.ones(len(z)), z])

    def perdida(beta):
        eta = x @ beta
        return np.sum(np.logaddexp(0, eta) - y * eta)

    ajuste = minimize(perdida, np.array([0.0, 1.0]), method="BFGS")
    if not ajuste.success and not np.isfinite(ajuste.fun):
        raise RuntimeError("No se pudo estimar la pendiente de calibracion.")
    return float(ajuste.x[0]), float(ajuste.x[1])

def metricas(y, p):
    intercepto = calibracion_intercepto(y, p)
    alfa, pendiente = calibracion_pendiente(y, p)
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, np.repeat(np.mean(y), len(y)))
    return {
        "prevalencia": float(np.mean(y)),
        "probabilidad_media": float(np.mean(p)),
        "diferencia_media_probabilidad": float(np.mean(p) - np.mean(y)),
        "calibracion_intercepto": intercepto,
        "intercepto_y_pendiente_libres": alfa,
        "calibracion_pendiente": pendiente,
        "AUC": float(roc_auc_score(y, p)),
        "Brier": float(brier),
        "Brier_referencia": float(brier_ref),
        "Brier_skill_score": float(1 - brier / brier_ref)
    }

def indices_bootstrap_estudiante(datos, rng):
    grupos = datos.reset_index(drop=True).groupby(ID).indices
    estudiantes = np.array(list(grupos.keys()), dtype=object)
    elegidos = rng.choice(estudiantes, size=len(estudiantes), replace=True)
    return np.concatenate([
        np.asarray(grupos[e], dtype=int) for e in elegidos
    ])

filas_metricas = []
filas_deciles = []
filas_bootstrap = []
rng = np.random.default_rng(SEMILLA)

for periodo, datos in [("2014B", prueba_b), ("2014J", prueba_j)]:
    x_test = preparar(datos).reindex(columns=x_train.columns, fill_value=0)
    prob_base = modelo.predict_proba(x_test)[:, 1]
    prob = calibrador.predict_proba(prob_base.reshape(-1, 1))[:, 1]
    y = datos[OBJETIVO].astype(int).to_numpy()

    resumen = metricas(y, prob)
    resumen.update({
        "periodo": periodo,
        "matriculas": len(datos),
        "estudiantes_unicos": datos[ID].nunique(),
        "retiros": int(y.sum())
    })
    filas_metricas.append(resumen)

    # Curva de calibracion en deciles de probabilidad predicha.
    curva = pd.DataFrame({"probabilidad": prob, "resultado": y})
    curva["decil"] = pd.qcut(
        curva["probabilidad"], q=10, labels=False, duplicates="drop"
    )
    tabla = curva.groupby("decil", observed=True).agg(
        matriculas=("resultado", "size"),
        probabilidad_media=("probabilidad", "mean"),
        tasa_retiro_observada=("resultado", "mean")
    ).reset_index()
    tabla["periodo"] = periodo
    filas_deciles.append(tabla)

    # Bootstrap pareado por estudiante: conserva todas sus matriculas.
    muestras = {
        "calibracion_intercepto": [],
        "calibracion_pendiente": [],
        "Brier": [],
        "Brier_skill_score": []
    }

    for replica in range(REPLICAS):
        idx = indices_bootstrap_estudiante(datos, rng)
        y_rep = y[idx]
        p_rep = prob[idx]

        if len(np.unique(y_rep)) < 2:
            continue

        m = metricas(y_rep, p_rep)
        for nombre in muestras:
            muestras[nombre].append(m[nombre])

    for nombre, valores in muestras.items():
        valores = np.asarray(valores, dtype=float)
        filas_bootstrap.append({
            "periodo": periodo,
            "indicador": nombre,
            "estimacion": resumen[nombre],
            "IC95_inferior": float(np.quantile(valores, 0.025)),
            "IC95_superior": float(np.quantile(valores, 0.975)),
            "replicas_validas": len(valores)
        })

# 4. Guardar resultados nuevos sin sobrescribir resultados anteriores.
pd.DataFrame(filas_metricas).to_csv(
    SALIDA / "152_metricas_calibracion_estricta.csv", index=False
)
pd.concat(filas_deciles, ignore_index=True).to_csv(
    SALIDA / "152_curva_calibracion_deciles.csv", index=False
)
pd.DataFrame(filas_bootstrap).to_csv(
    SALIDA / "152_intervalos_bootstrap_calibracion.csv", index=False
)

print("\nEXPERIMENTO 152: DIAGNOSTICO DE CALIBRACION ESTRICTO")
print("Cohortes independientes: comprobadas.")
print("Modelo: combinado sin identidad del modulo, reproduccion del experimento 150.")
print("Bootstrap por estudiante:", REPLICAS, "replicas solicitadas.")

print("\nMETRICAS DE CALIBRACION")
print(pd.DataFrame(filas_metricas).round(5).to_string(index=False))

print("\nCURVA DE CALIBRACION POR DECILES")
print(pd.concat(filas_deciles, ignore_index=True).round(5).to_string(index=False))

print("\nINTERVALOS BOOTSTRAP")
print(pd.DataFrame(filas_bootstrap).round(5).to_string(index=False))

print("\nArchivos guardados:")
print(SALIDA / "152_metricas_calibracion_estricta.csv")
print(SALIDA / "152_curva_calibracion_deciles.csv")
print(SALIDA / "152_intervalos_bootstrap_calibracion.csv")
