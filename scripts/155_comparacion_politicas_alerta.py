"""Experimento 155: compara umbrales fijos con ranking top-k para capacidad operativa.

Replica las cohortes, modelos y calibracion del experimento 150.
La politica top-k selecciona exactamente floor(n * capacidad) matriculas
con mayor probabilidad, sin consultar las etiquetas de la cohorte de prueba.
No modifica datos ni resultados anteriores.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, brier_score_loss
)

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "interim" / "week4_combined_features_v4.parquet"
SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

OBJETIVO = "target_future_withdrawal"
ID = "id_student"
PRESENTACION = "code_presentation"
VARIABLES_ESTUDIANTE = [
    "week3_clicks", "week4_clicks", "week4_active_days",
    "week4_activity_records", "has_week4_activity",
    "weekly_change", "clics_recientes_3_4",
    "evaluaciones_entregadas_34"
]
VARIABLES_CONTEXTO = ["evaluaciones_programadas_34"]
CAPACIDADES = [0.20, 0.30, 0.40]
CLAVE = ["code_module", "code_presentation", ID]


def preparar(datos, variables):
    x = datos[variables].copy()
    for columna in variables:
        x[columna] = pd.to_numeric(
            x[columna], errors="coerce"
        ).fillna(0)
    return x


def elegir_umbral(probabilidades, tasa_objetivo):
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


def prediccion_top_k(datos, probabilidades, tasa_objetivo):
    """Marca como alerta las k probabilidades mas altas.

    k=floor(n*tasa): nunca supera la capacidad porcentual solicitada.
    Los empates se resuelven por modulo, presentacion e id_student.
    """
    n = len(datos)
    k = int(np.floor(n * tasa_objetivo))
    pred = np.zeros(n, dtype=int)

    if k == 0:
        return pred, k

    orden = datos[CLAVE].copy().reset_index(drop=True)
    orden["_probabilidad"] = np.asarray(probabilidades)
    orden["_posicion_original"] = np.arange(n)
    orden = orden.sort_values(
        ["_probabilidad", "code_module", "code_presentation", ID],
        ascending=[False, True, True, True],
        kind="mergesort"
    )
    posiciones = orden["_posicion_original"].to_numpy()[:k]
    pred[posiciones] = 1
    return pred, k


def metricas(nombre, periodo, datos, prob, pred, capacidad,
             politica, umbral=np.nan, k=np.nan):
    y = datos[OBJETIVO].astype(int).to_numpy()
    return {
        "modelo": nombre,
        "periodo": periodo,
        "politica": politica,
        "capacidad_objetivo": capacidad,
        "matriculas": len(datos),
        "estudiantes_unicos": datos[ID].nunique(),
        "retiros": int(y.sum()),
        "alertas": int(pred.sum()),
        "porcentaje_alertas": float(pred.mean() * 100),
        "umbral_aplicado": umbral,
        "k_alertas_top_k": k,
        "AUC": roc_auc_score(y, prob),
        "F1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "Brier": brier_score_loss(y, prob)
    }


def main():
    df = pd.read_parquet(ARCHIVO)
    necesarias = (
        VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
        + [OBJETIVO, ID, PRESENTACION, "code_module"]
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

    p13b, p13j = periodos["2013B"], periodos["2013J"]
    p14b, p14j = periodos["2014B"], periodos["2014J"]
    ids_13b, ids_13j = set(p13b[ID]), set(p13j[ID])
    ids_14b = set(p14b[ID])

    divisor = StratifiedGroupKFold(
        n_splits=2, shuffle=True, random_state=42
    )
    i_cal, i_umbral = next(
        divisor.split(p13j, p13j[OBJETIVO], groups=p13j[ID])
    )
    calibracion = p13j.iloc[i_cal].copy()
    seleccion = p13j.iloc[i_umbral].copy()
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
        "seleccion_umbral": seleccion,
        "prueba_2014B": prueba_b,
        "prueba_2014J": prueba_j
    }
    conjuntos_ids = {n: set(d[ID]) for n, d in cohortes.items()}
    nombres = list(conjuntos_ids)
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            if conjuntos_ids[nombres[i]] & conjuntos_ids[nombres[j]]:
                raise AssertionError(
                    "Hay estudiantes compartidos entre "
                    + nombres[i] + " y " + nombres[j]
                )
    for nombre, datos in cohortes.items():
        if datos.empty or datos[OBJETIVO].nunique() != 2:
            raise ValueError("Cohorte vacia o sin ambas clases: " + nombre)
        if datos.duplicated(CLAVE).any():
            raise ValueError("Claves de matricula duplicadas en " + nombre)

    especificaciones = {
        "estudiante_solo": VARIABLES_ESTUDIANTE,
        "contexto_curso_solo": VARIABLES_CONTEXTO,
        "combinado_sin_identidad_modulo":
            VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
    }

    filas = []
    filas_seleccion = []

    for nombre, variables in especificaciones.items():
        x_train = preparar(entrenamiento, variables)
        x_cal = preparar(calibracion, variables).reindex(
            columns=x_train.columns, fill_value=0
        )
        x_sel = preparar(seleccion, variables).reindex(
            columns=x_train.columns, fill_value=0
        )

        modelo = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
        )
        modelo.fit(x_train, entrenamiento[OBJETIVO].astype(int))

        prob_cal_base = modelo.predict_proba(x_cal)[:, 1]
        calibrador = LogisticRegression(max_iter=1000)
        calibrador.fit(
            prob_cal_base.reshape(-1, 1),
            calibracion[OBJETIVO].astype(int)
        )

        prob_sel_base = modelo.predict_proba(x_sel)[:, 1]
        prob_sel = calibrador.predict_proba(
            prob_sel_base.reshape(-1, 1)
        )[:, 1]

        for capacidad in CAPACIDADES:
            umbral, tasa_sel = elegir_umbral(prob_sel, capacidad)
            pred_sel_umbral = (prob_sel >= umbral).astype(int)
            pred_sel_top, k_sel = prediccion_top_k(
                seleccion, prob_sel, capacidad
            )
            filas_seleccion.append({
                "modelo": nombre,
                "capacidad_objetivo": capacidad,
                "umbral": umbral,
                "alertas_umbral_seleccion": int(pred_sel_umbral.sum()),
                "porcentaje_umbral_seleccion": pred_sel_umbral.mean() * 100,
                "alertas_top_k_seleccion": int(pred_sel_top.sum()),
                "k_top_k_seleccion": k_sel,
                "F1_umbral_seleccion": f1_score(
                    seleccion[OBJETIVO].astype(int), pred_sel_umbral,
                    zero_division=0
                ),
                "F1_top_k_seleccion": f1_score(
                    seleccion[OBJETIVO].astype(int), pred_sel_top,
                    zero_division=0
                )
            })

            for periodo, datos in [("2014B", prueba_b), ("2014J", prueba_j)]:
                x_test = preparar(datos, variables).reindex(
                    columns=x_train.columns, fill_value=0
                )
                prob_base = modelo.predict_proba(x_test)[:, 1]
                prob_test = calibrador.predict_proba(
                    prob_base.reshape(-1, 1)
                )[:, 1]

                pred_umbral = (prob_test >= umbral).astype(int)
                pred_top, k_test = prediccion_top_k(
                    datos, prob_test, capacidad
                )

                filas.append(metricas(
                    nombre, periodo, datos, prob_test, pred_umbral,
                    capacidad, "umbral_fijo", umbral=umbral
                ))
                filas.append(metricas(
                    nombre, periodo, datos, prob_test, pred_top,
                    capacidad, "ranking_top_k", k=k_test
                ))

    resultados = pd.DataFrame(filas)
    seleccion_resultados = pd.DataFrame(filas_seleccion)
    ruta_resultados = SALIDA / "155_comparacion_politicas_alerta.csv"
    ruta_seleccion = SALIDA / "155_politicas_en_seleccion.csv"
    resultados.to_csv(ruta_resultados, index=False)
    seleccion_resultados.to_csv(ruta_seleccion, index=False)

    print("\nEXPERIMENTO 155: UMBRAL FIJO FRENTE A RANKING TOP-K")
    print("\nCOHORTES ESTRICTAMENTE INDEPENDIENTES")
    for nombre, datos in cohortes.items():
        print(
            nombre, "| matriculas:", len(datos),
            "| estudiantes:", datos[ID].nunique(),
            "| retiros:", int(datos[OBJETIVO].sum())
        )
    print("\nPOLITICAS EN LA COHORTE DE SELECCION")
    print(seleccion_resultados.round(4).to_string(index=False))
    print("\nRESULTADOS EN PRUEBAS INDEPENDIENTES")
    print(resultados.round(4).to_string(index=False))
    print("\nARCHIVOS GUARDADOS:")
    print(ruta_resultados)
    print(ruta_seleccion)
    print("\nNOTA: top-k fija el numero de alertas sin usar etiquetas de prueba.")
    print("El AUC y Brier son iguales entre politicas porque usan las mismas probabilidades.")


if __name__ == "__main__":
    main()
