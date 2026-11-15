import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.special import expit, logit
from scipy.optimize import brentq, minimize

# =====================================================
# EXPERIMENTO 153:
# ACTUALIZACION TEMPORAL DEL CALIBRADOR
#
# Modelo base: entrenamiento 2013B.
# Calibrador inicial: mitad de 2013J.
# Actualizacion del calibrador: 2014B.
# Evaluacion final: estudiantes nuevos de 2014J.
#
# 2014B se usa para actualizar el calibrador, por lo que
# NO se presenta como prueba independiente de ese calibrador.
# =====================================================

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

OBJETIVO = "target_future_withdrawal"
ID = "id_student"
PRESENTACION = "code_presentation"
SEMILLA = 153
REPLICAS = 1000

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

def preparar(datos):
    x = datos[VARIABLES].copy()
    for c in VARIABLES:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    return x

def intercepto_calibracion(y, p):
    z = logit(np.clip(p, 1e-10, 1 - 1e-10))
    return float(brentq(
        lambda a: np.mean(expit(z + a)) - np.mean(y),
        -30, 30
    ))

def pendiente_calibracion(y, p):
    z = logit(np.clip(p, 1e-10, 1 - 1e-10))
    x = np.column_stack([np.ones(len(z)), z])

    def perdida(beta):
        eta = x @ beta
        return np.sum(np.logaddexp(0, eta) - y * eta)

    ajuste = minimize(perdida, np.array([0.0, 1.0]), method="BFGS")
    if not np.isfinite(ajuste.fun):
        raise RuntimeError("No se pudo estimar la pendiente.")
    return float(ajuste.x[1])

def metricas(y, p):
    brier = brier_score_loss(y, p)
    referencia = brier_score_loss(y, np.repeat(np.mean(y), len(y)))
    return {
        "prevalencia": float(np.mean(y)),
        "probabilidad_media": float(np.mean(p)),
        "diferencia_media_probabilidad": float(np.mean(p) - np.mean(y)),
        "intercepto_calibracion": intercepto_calibracion(y, p),
        "pendiente_calibracion": pendiente_calibracion(y, p),
        "AUC": float(roc_auc_score(y, p)),
        "Brier": float(brier),
        "Brier_referencia": float(referencia),
        "Brier_skill_score": float(1 - brier / referencia)
    }

def indices_bootstrap(datos, rng):
    grupos = datos.reset_index(drop=True).groupby(ID).indices
    estudiantes = np.array(list(grupos.keys()), dtype=object)
    elegidos = rng.choice(estudiantes, size=len(estudiantes), replace=True)
    return np.concatenate([np.asarray(grupos[e], dtype=int) for e in elegidos])

# 1. Reconstruir exactamente las cohortes del experimento 150.
df = pd.read_parquet(ARCHIVO)
periodos = {
    n: df[df[PRESENTACION] == n].copy()
    for n in ["2013B", "2013J", "2014B", "2014J"]
}
p13b, p13j = periodos["2013B"], periodos["2013J"]
p14b, p14j = periodos["2014B"], periodos["2014J"]

ids_13b = set(p13b[ID])
ids_13j = set(p13j[ID])
ids_14b = set(p14b[ID])

divisor = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
i_cal, i_sel = next(
    divisor.split(p13j, p13j[OBJETIVO], groups=p13j[ID])
)
calibracion_inicial = p13j.iloc[i_cal].copy()
seleccion_umbral = p13j.iloc[i_sel].copy()
entrenamiento = p13b[~p13b[ID].isin(ids_13j)].copy()
actualizacion = p14b[
    ~p14b[ID].isin(ids_13b.union(ids_13j))
].copy()
prueba = p14j[
    ~p14j[ID].isin(ids_13b.union(ids_13j).union(ids_14b))
].copy()

cohortes = {
    "entrenamiento": entrenamiento,
    "calibracion_inicial": calibracion_inicial,
    "seleccion_umbral": seleccion_umbral,
    "actualizacion_2014B": actualizacion,
    "prueba_final_2014J": prueba
}
ids = {n: set(d[ID]) for n, d in cohortes.items()}
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

# 2. Entrenar el mismo modelo base combinado del experimento 150.
x_train = preparar(entrenamiento)
modelo = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
)
modelo.fit(x_train, entrenamiento[OBJETIVO].astype(int))

# 3. Ajustar calibrador inicial usando solamente calibracion 2013J.
p_cal_base = modelo.predict_proba(preparar(calibracion_inicial))[:, 1]
calibrador_inicial = LogisticRegression(max_iter=1000)
calibrador_inicial.fit(
    p_cal_base.reshape(-1, 1),
    calibracion_inicial[OBJETIVO].astype(int)
)

# 4. Obtener probabilidades base para actualizacion y prueba final.
p_2014b_base = modelo.predict_proba(preparar(actualizacion))[:, 1]
p_2014j_base = modelo.predict_proba(preparar(prueba))[:, 1]

# Actualizacion del calibrador con 2014B, sin reentrenar el modelo base.
calibrador_actualizado = LogisticRegression(max_iter=1000)
calibrador_actualizado.fit(
    p_2014b_base.reshape(-1, 1),
    actualizacion[OBJETIVO].astype(int)
)

# Comparar ambos calibradores en la misma prueba independiente 2014J.
p_original = calibrador_inicial.predict_proba(
    p_2014j_base.reshape(-1, 1)
)[:, 1]
p_actualizado = calibrador_actualizado.predict_proba(
    p_2014j_base.reshape(-1, 1)
)[:, 1]
y_test = prueba[OBJETIVO].astype(int).to_numpy()

# 5. Metricas de la evaluacion final.
filas = []
for nombre, prob in [
    ("calibrador_2013J_original", p_original),
    ("calibrador_actualizado_2014B", p_actualizado)
]:
    m = metricas(y_test, prob)
    m.update({
        "calibrador": nombre,
        "periodo_evaluacion": "2014J",
        "matriculas": len(prueba),
        "estudiantes_unicos": prueba[ID].nunique(),
        "retiros": int(y_test.sum())
    })
    filas.append(m)

# Diagnostico descriptivo de ajuste del calibrador en 2014B.
# Es in-sample para el calibrador actualizado y NO es una prueba independiente.
p_2014b_ajustado = calibrador_actualizado.predict_proba(
    p_2014b_base.reshape(-1, 1)
)[:, 1]
m_ajuste = metricas(
    actualizacion[OBJETIVO].astype(int).to_numpy(),
    p_2014b_ajustado
)
m_ajuste.update({
    "calibrador": "calibrador_actualizado_2014B",
    "periodo_evaluacion": "2014B_ajuste_no_independiente",
    "matriculas": len(actualizacion),
    "estudiantes_unicos": actualizacion[ID].nunique(),
    "retiros": int(actualizacion[OBJETIVO].sum())
})
filas.append(m_ajuste)

# 6. Curvas por deciles sobre 2014J.
curvas = []
for nombre, prob in [
    ("calibrador_2013J_original", p_original),
    ("calibrador_actualizado_2014B", p_actualizado)
]:
    d = pd.DataFrame({"probabilidad": prob, "resultado": y_test})
    d["decil"] = pd.qcut(
        d["probabilidad"], q=10, labels=False, duplicates="drop"
    )
    t = d.groupby("decil", observed=True).agg(
        matriculas=("resultado", "size"),
        probabilidad_media=("probabilidad", "mean"),
        tasa_retiro_observada=("resultado", "mean")
    ).reset_index()
    t["calibrador"] = nombre
    t["periodo"] = "2014J"
    curvas.append(t)

# 7. Bootstrap pareado por estudiante en la prueba final.
# Los dos calibradores se comparan en las mismas muestras de estudiantes.
rng = np.random.default_rng(SEMILLA)
indicadores = [
    "Brier", "intercepto_calibracion", "pendiente_calibracion",
    "Brier_skill_score"
]
boot = {n: {"original": [], "actualizado": [], "diferencia": []}
        for n in indicadores}

for replica in range(REPLICAS):
    idx = indices_bootstrap(prueba, rng)
    y_rep = y_test[idx]
    if len(np.unique(y_rep)) < 2:
        continue

    m0 = metricas(y_rep, p_original[idx])
    m1 = metricas(y_rep, p_actualizado[idx])

    for indicador in indicadores:
        boot[indicador]["original"].append(m0[indicador])
        boot[indicador]["actualizado"].append(m1[indicador])
        boot[indicador]["diferencia"].append(
            m1[indicador] - m0[indicador]
        )

filas_boot = []
for indicador in indicadores:
    for tipo, valores in boot[indicador].items():
        v = np.asarray(valores, dtype=float)
        filas_boot.append({
            "indicador": indicador,
            "comparacion": tipo,
            "estimacion_media_bootstrap": float(np.mean(v)),
            "IC95_inferior": float(np.quantile(v, 0.025)),
            "IC95_superior": float(np.quantile(v, 0.975)),
            "replicas_validas": len(v)
        })

# 8. Guardar salidas independientes.
pd.DataFrame(filas).to_csv(
    SALIDA / "153_comparacion_actualizacion_calibrador.csv", index=False
)
pd.concat(curvas, ignore_index=True).to_csv(
    SALIDA / "153_curvas_calibracion_2014J.csv", index=False
)
pd.DataFrame(filas_boot).to_csv(
    SALIDA / "153_bootstrap_comparacion_calibradores.csv", index=False
)

print("\nEXPERIMENTO 153: ACTUALIZACION TEMPORAL DEL CALIBRADOR")
print("Modelo base entrenado en 2013B; calibrador inicial ajustado en 2013J.")
print("Calibrador actualizado con 2014B; evaluacion independiente en 2014J.")
print("Estudiantes compartidos entre cohortes: 0.")
print("Bootstrap pareado por estudiante:", REPLICAS, "replicas solicitadas.")

print("\nCOMPARACION DE CALIBRADORES")
print(pd.DataFrame(filas).round(5).to_string(index=False))

print("\nCURVAS DE CALIBRACION EN 2014J")
print(pd.concat(curvas, ignore_index=True).round(5).to_string(index=False))

print("\nBOOTSTRAP PAREADO: DIFERENCIAS = ACTUALIZADO - ORIGINAL")
print(pd.DataFrame(filas_boot).round(5).to_string(index=False))

print("\nArchivos guardados:")
print(SALIDA / "153_comparacion_actualizacion_calibrador.csv")
print(SALIDA / "153_curvas_calibracion_2014J.csv")
print(SALIDA / "153_bootstrap_comparacion_calibradores.csv")
