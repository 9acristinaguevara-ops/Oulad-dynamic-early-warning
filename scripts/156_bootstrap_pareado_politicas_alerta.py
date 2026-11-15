"""Experimento 156: bootstrap pareado por estudiante para comparar politicas de alerta.

Reproduce el modelo/calibracion del experimento 150 y compara, en 2014B/2014J,
umbral fijo frente a ranking top-k. El bootstrap remuestrea estudiantes completos
y mantiene fijas las predicciones generadas en la cohorte original; por tanto,
los intervalos son condicionales al modelo, calibrador, umbral y ranking ya ajustados.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

OBJETIVO = "target_future_withdrawal"
ID = "id_student"
PRESENTACION = "code_presentation"
CLAVE = ["code_module", "code_presentation", ID]
VARIABLES_ESTUDIANTE = [
    "week3_clicks", "week4_clicks", "week4_active_days",
    "week4_activity_records", "has_week4_activity",
    "weekly_change", "clics_recientes_3_4",
    "evaluaciones_entregadas_34"
]
VARIABLES_CONTEXTO = ["evaluaciones_programadas_34"]
CAPACIDADES = [0.20, 0.30, 0.40]
REPLICAS = 2000
SEMILLA = 156


def preparar(datos, variables):
    x = datos[variables].copy()
    for columna in variables:
        x[columna] = pd.to_numeric(
            x[columna], errors="coerce"
        ).fillna(0)
    return x


def elegir_umbral(probabilidades, tasa_objetivo):
    candidatos = np.unique(probabilidades)
    mejor = candidatos[0]
    diferencia_minima = float("inf")
    for candidato in candidatos:
        diferencia = abs(
            (probabilidades >= candidato).mean() - tasa_objetivo
        )
        if diferencia < diferencia_minima:
            diferencia_minima = diferencia
            mejor = candidato
    return float(mejor)


def top_k(datos, probabilidades, capacidad):
    n = len(datos)
    k = int(np.floor(n * capacidad))
    pred = np.zeros(n, dtype=int)
    if k == 0:
        return pred
    orden = datos[CLAVE].copy().reset_index(drop=True)
    orden["_prob"] = np.asarray(probabilidades)
    orden["_pos"] = np.arange(n)
    orden = orden.sort_values(
        ["_prob", "code_module", "code_presentation", ID],
        ascending=[False, True, True, True], kind="mergesort"
    )
    pred[orden["_pos"].to_numpy()[:k]] = 1
    return pred


def metricas(y, pred):
    return {
        "F1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0)
    }


def main():
    df = pd.read_parquet(ARCHIVO)
    requeridas = (
        VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
        + [OBJETIVO, ID, PRESENTACION, "code_module"]
    )
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise ValueError("Faltan columnas: " + ", ".join(faltan))

    periodos = {
        p: df[df[PRESENTACION] == p].copy()
        for p in ["2013B", "2013J", "2014B", "2014J"]
    }
    p13b, p13j = periodos["2013B"], periodos["2013J"]
    p14b, p14j = periodos["2014B"], periodos["2014J"]
    ids_13b, ids_13j = set(p13b[ID]), set(p13j[ID])
    ids_14b = set(p14b[ID])

    divisor = StratifiedGroupKFold(
        n_splits=2, shuffle=True, random_state=42
    )
    i_cal, i_sel = next(
        divisor.split(p13j, p13j[OBJETIVO], groups=p13j[ID])
    )
    calibracion = p13j.iloc[i_cal].copy()
    seleccion = p13j.iloc[i_sel].copy()
    entrenamiento = p13b[~p13b[ID].isin(ids_13j)].copy()
    pruebas = {
        "2014B": p14b[
            ~p14b[ID].isin(ids_13b.union(ids_13j))
        ].copy(),
        "2014J": p14j[
            ~p14j[ID].isin(ids_13b.union(ids_13j).union(ids_14b))
        ].copy()
    }

    cohortes = {
        "entrenamiento": entrenamiento,
        "calibracion": calibracion,
        "seleccion": seleccion,
        **{"prueba_" + p: d for p, d in pruebas.items()}
    }
    id_sets = {n: set(d[ID]) for n, d in cohortes.items()}
    nombres = list(id_sets)
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            if id_sets[nombres[i]] & id_sets[nombres[j]]:
                raise AssertionError("Solapamiento de estudiantes entre cohortes.")

    especificaciones = {
        "estudiante_solo": VARIABLES_ESTUDIANTE,
        "contexto_curso_solo": VARIABLES_CONTEXTO,
        "combinado_sin_identidad_modulo":
            VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
    }

    rng = np.random.default_rng(SEMILLA)
    filas = []

    for nombre, variables in especificaciones.items():
        x_train = preparar(entrenamiento, variables)
        modelo = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
        )
        modelo.fit(x_train, entrenamiento[OBJETIVO].astype(int))

        x_cal = preparar(calibracion, variables).reindex(
            columns=x_train.columns, fill_value=0
        )
        prob_cal_base = modelo.predict_proba(x_cal)[:, 1]
        calibrador = LogisticRegression(max_iter=1000)
        calibrador.fit(
            prob_cal_base.reshape(-1, 1),
            calibracion[OBJETIVO].astype(int)
        )

        x_sel = preparar(seleccion, variables).reindex(
            columns=x_train.columns, fill_value=0
        )
        prob_sel_base = modelo.predict_proba(x_sel)[:, 1]
        prob_sel = calibrador.predict_proba(
            prob_sel_base.reshape(-1, 1)
        )[:, 1]

        umbrales = {
            cap: elegir_umbral(prob_sel, cap) for cap in CAPACIDADES
        }

        for periodo, datos in pruebas.items():
            x_test = preparar(datos, variables).reindex(
                columns=x_train.columns, fill_value=0
            )
            prob_base = modelo.predict_proba(x_test)[:, 1]
            prob = calibrador.predict_proba(
                prob_base.reshape(-1, 1)
            )[:, 1]
            y = datos[OBJETIVO].astype(int).to_numpy()
            predicciones = {}

            for cap in CAPACIDADES:
                predicciones[(cap, "umbral_fijo")] = (
                    prob >= umbrales[cap]
                ).astype(int)
                predicciones[(cap, "ranking_top_k")] = top_k(
                    datos, prob, cap
                )

            # Guardar metricas puntuales y bootstrap pareado por estudiante.
            estudiantes = datos[ID].drop_duplicates().to_numpy()
            indices_por_estudiante = {
                estudiante: np.flatnonzero(
                    datos[ID].to_numpy() == estudiante
                )
                for estudiante in estudiantes
            }

            for cap in CAPACIDADES:
                pred_umbral = predicciones[(cap, "umbral_fijo")]
                pred_top = predicciones[(cap, "ranking_top_k")]
                m_umbral = metricas(y, pred_umbral)
                m_top = metricas(y, pred_top)

                muestras = {
                    "F1": [], "precision": [], "recall": []
                }
                validas = 0
                for _ in range(REPLICAS):
                    seleccion_ids = rng.choice(
                        estudiantes, size=len(estudiantes), replace=True
                    )
                    indices = np.concatenate([
                        indices_por_estudiante[e] for e in seleccion_ids
                    ])
                    y_b = y[indices]
                    if np.unique(y_b).size < 2:
                        continue
                    validas += 1
                    a = metricas(y_b, pred_umbral[indices])
                    b = metricas(y_b, pred_top[indices])
                    for metrica in muestras:
                        muestras[metrica].append(
                            b[metrica] - a[metrica]
                        )

                for metrica in muestras:
                    valores = np.asarray(muestras[metrica])
                    filas.append({
                        "modelo": nombre,
                        "periodo": periodo,
                        "capacidad": cap,
                        "metrica": metrica,
                        "umbral_fijo": m_umbral[metrica],
                        "ranking_top_k": m_top[metrica],
                        "diferencia_top_k_menos_umbral": (
                            m_top[metrica] - m_umbral[metrica]
                        ),
                        "IC95_inferior": (
                            np.quantile(valores, 0.025)
                            if len(valores) else np.nan
                        ),
                        "IC95_superior": (
                            np.quantile(valores, 0.975)
                            if len(valores) else np.nan
                        ),
                        "replicas_validas": validas,
                        "replicas_solicitadas": REPLICAS,
                        "estudiantes_bootstrap": len(estudiantes),
                        "matriculas_prueba": len(datos)
                    })

    resultados = pd.DataFrame(filas)
    ruta = SALIDA / "156_bootstrap_pareado_politicas_alerta.csv"
    resultados.to_csv(ruta, index=False)

    print("\nEXPERIMENTO 156: BOOTSTRAP PAREADO POR ESTUDIANTE")
    print("Replicas solicitadas:", REPLICAS)
    print("La diferencia es top-k menos umbral fijo.")
    print("IC95 que no cruza cero indica diferencia compatible con efecto distinto de cero.")
    print("\nRESULTADOS")
    print(resultados.round(5).to_string(index=False))
    print("\nARCHIVO GUARDADO:")
    print(ruta)
    print("\nNOTA: intervalos condicionales a modelos, calibradores, umbrales y rankings ajustados.")


if __name__ == "__main__":
    main()
