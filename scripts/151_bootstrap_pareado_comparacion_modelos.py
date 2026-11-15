import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

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

REPLICAS = 1000
SEMILLA = 151

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

# =====================================================
# 2. CARGAR Y RECONSTRUIR LAS COHORTES ESTRICTAS
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
            raise AssertionError("Hay estudiantes compartidos entre cohortes.")

# =====================================================
# 3. PREPARAR VARIABLES Y MODELOS
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

predicciones = {}
y_train = entrenamiento[OBJETIVO].astype(int)
y_cal = calibracion[OBJETIVO].astype(int)

for nombre, variables in especificaciones.items():
    x_train = preparar(entrenamiento, variables)
    x_cal = preparar(calibracion, variables).reindex(
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

    predicciones[nombre] = {}

    for periodo, datos in [
        ("2014B", prueba_b), ("2014J", prueba_j)
    ]:
        x_test = preparar(datos, variables).reindex(
            columns=x_train.columns, fill_value=0
        )
        prob_base = modelo.predict_proba(x_test)[:, 1]
        prob = calibrador.predict_proba(
            prob_base.reshape(-1, 1)
        )[:, 1]

        predicciones[nombre][periodo] = prob

# =====================================================
# 4. BOOTSTRAP PAREADO POR ESTUDIANTE
# =====================================================

def indices_bootstrap_por_estudiante(datos, rng):
    # Devuelve indices posicionales y conserva juntas todas
    # las matriculas de cada estudiante seleccionado.
    grupos = datos.reset_index(drop=True).groupby(ID).indices
    estudiantes = np.array(list(grupos.keys()), dtype=object)
    seleccionados = rng.choice(
        estudiantes, size=len(estudiantes), replace=True
    )

    partes = []
    for estudiante in seleccionados:
        partes.append(np.asarray(grupos[estudiante], dtype=int))

    return np.concatenate(partes)

def intervalo(valores):
    return (
        float(np.quantile(valores, 0.025)),
        float(np.quantile(valores, 0.975))
    )

filas = []
comparaciones = [
    ("combinado_sin_identidad_modulo", "estudiante_solo"),
    ("combinado_sin_identidad_modulo", "contexto_curso_solo")
]

rng = np.random.default_rng(SEMILLA)

for periodo, datos_originales in [
    ("2014B", prueba_b), ("2014J", prueba_j)
]:
    datos = datos_originales.reset_index(drop=True)
    y = datos[OBJETIVO].astype(int).to_numpy()

    if len(np.unique(y)) != 2:
        raise ValueError("La cohorte de prueba no tiene ambas clases.")

    aucs = {
        nombre: roc_auc_score(y, predicciones[nombre][periodo])
        for nombre in especificaciones
    }
    briers = {
        nombre: brier_score_loss(y, predicciones[nombre][periodo])
        for nombre in especificaciones
    }

    # Guardar intervalos individuales y diferencias pareadas.
    muestras_auc = {nombre: [] for nombre in especificaciones}
    muestras_brier = {nombre: [] for nombre in especificaciones}
    muestras_diferencias = {
        (a, b): {"auc": [], "brier": []}
        for a, b in comparaciones
    }

    for replica in range(REPLICAS):
        indices = indices_bootstrap_por_estudiante(datos, rng)
        y_rep = y[indices]

        if len(np.unique(y_rep)) < 2:
            continue

        prob_rep = {
            nombre: predicciones[nombre][periodo][indices]
            for nombre in especificaciones
        }

        auc_rep = {}
        brier_rep = {}

        for nombre in especificaciones:
            auc_rep[nombre] = roc_auc_score(y_rep, prob_rep[nombre])
            brier_rep[nombre] = brier_score_loss(
                y_rep, prob_rep[nombre]
            )
            muestras_auc[nombre].append(auc_rep[nombre])
            muestras_brier[nombre].append(brier_rep[nombre])

        for a, b in comparaciones:
            muestras_diferencias[(a, b)]["auc"].append(
                auc_rep[a] - auc_rep[b]
            )
            muestras_diferencias[(a, b)]["brier"].append(
                brier_rep[a] - brier_rep[b]
            )

    for nombre in especificaciones:
        auc_inf, auc_sup = intervalo(muestras_auc[nombre])
        brier_inf, brier_sup = intervalo(muestras_brier[nombre])

        filas.append({
            "periodo": periodo,
            "tipo_resultado": "intervalo_modelo",
            "modelo": nombre,
            "comparador": "",
            "AUC": aucs[nombre],
            "AUC_IC95_inf": auc_inf,
            "AUC_IC95_sup": auc_sup,
            "Brier": briers[nombre],
            "Brier_IC95_inf": brier_inf,
            "Brier_IC95_sup": brier_sup,
            "diferencia_AUC": np.nan,
            "diferencia_AUC_IC95_inf": np.nan,
            "diferencia_AUC_IC95_sup": np.nan,
            "prob_diferencia_AUC_mayor_0": np.nan,
            "diferencia_Brier": np.nan,
            "diferencia_Brier_IC95_inf": np.nan,
            "diferencia_Brier_IC95_sup": np.nan,
            "prob_diferencia_Brier_menor_0": np.nan,
            "replicas_validas": len(muestras_auc[nombre])
        })

    for a, b in comparaciones:
        dif_auc = np.asarray(muestras_diferencias[(a, b)]["auc"])
        dif_brier = np.asarray(muestras_diferencias[(a, b)]["brier"])
        auc_inf, auc_sup = intervalo(dif_auc)
        brier_inf, brier_sup = intervalo(dif_brier)

        filas.append({
            "periodo": periodo,
            "tipo_resultado": "diferencia_pareada",
            "modelo": a,
            "comparador": b,
            "AUC": aucs[a],
            "AUC_IC95_inf": np.nan,
            "AUC_IC95_sup": np.nan,
            "Brier": briers[a],
            "Brier_IC95_inf": np.nan,
            "Brier_IC95_sup": np.nan,
            "diferencia_AUC": aucs[a] - aucs[b],
            "diferencia_AUC_IC95_inf": auc_inf,
            "diferencia_AUC_IC95_sup": auc_sup,
            "prob_diferencia_AUC_mayor_0": float(
                np.mean(dif_auc > 0)
            ),
            "diferencia_Brier": briers[a] - briers[b],
            "diferencia_Brier_IC95_inf": brier_inf,
            "diferencia_Brier_IC95_sup": brier_sup,
            "prob_diferencia_Brier_menor_0": float(
                np.mean(dif_brier < 0)
            ),
            "replicas_validas": len(dif_auc)
        })

tabla = pd.DataFrame(filas)
archivo_salida = SALIDA / "151_bootstrap_pareado_comparacion_modelos.csv"
tabla.to_csv(archivo_salida, index=False)

# =====================================================
# 5. INFORME
# =====================================================

print("\nEXPERIMENTO 151: BOOTSTRAP PAREADO POR ESTUDIANTE")
print("Replicas solicitadas:", REPLICAS)
print("\nRESULTADOS: INTERVALOS Y DIFERENCIAS")
print(tabla.round(5).to_string(index=False))
print("\nARCHIVO GUARDADO:")
print(archivo_salida)
