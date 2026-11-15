"""Experimento 159: sensibilidad de top-k al desempate aleatorio.

Reproduce los modelos y cohortes de los experimentos 150 y 158.
En cada repeticion mantiene todas las filas por encima del corte y elige
aleatoriamente las plazas restantes entre las filas empatadas en el corte.
Esto mide sensibilidad al desempate, no incertidumbre de muestreo ni
incertidumbre por reentrenamiento.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

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
REPETICIONES = 2000
SEMILLA = 159


def preparar(datos, variables):
    x = datos[variables].copy()
    for columna in variables:
        x[columna] = pd.to_numeric(
            x[columna], errors="coerce"
        ).fillna(0)
    return x


def metricas(y, pred):
    return {
        "F1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0)
    }


def seleccionar_top_k(probabilidades, capacidad, rng=None):
    n = len(probabilidades)
    k = int(np.floor(n * capacidad))
    corte = np.sort(probabilidades)[::-1][k - 1]
    mayores = np.flatnonzero(probabilidades > corte)
    empatados = np.flatnonzero(probabilidades == corte)
    plazas = k - len(mayores)

    seleccion = np.zeros(n, dtype=int)
    seleccion[mayores] = 1

    if rng is None:
        # Reproduce el desempate estable por orden de fila del experimento 158.
        seleccion[empatados[:plazas]] = 1
    elif plazas > 0:
        elegidos = rng.choice(empatados, size=plazas, replace=False)
        seleccion[elegidos] = 1

    return seleccion, corte, len(empatados), plazas


def resumen(valores):
    a = np.asarray(valores, dtype=float)
    return float(np.mean(a)), float(np.quantile(a, 0.025)), float(
        np.quantile(a, 0.975)
    )


def main():
    df = pd.read_parquet(ARCHIVO)
    requeridas = (
        VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
        + [OBJETIVO, ID, PRESENTACION, "code_module"]
    )
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise ValueError("Faltan columnas: " + ", ".join(faltan))

    p13b = df[df[PRESENTACION] == "2013B"].copy()
    p13j = df[df[PRESENTACION] == "2013J"].copy()
    p14b = df[df[PRESENTACION] == "2014B"].copy()
    p14j = df[df[PRESENTACION] == "2014J"].copy()

    ids_13b, ids_13j = set(p13b[ID]), set(p13j[ID])
    ids_14b = set(p14b[ID])
    divisor = StratifiedGroupKFold(
        n_splits=2, shuffle=True, random_state=42
    )
    i_cal, _ = next(
        divisor.split(p13j, p13j[OBJETIVO], groups=p13j[ID])
    )
    calibracion = p13j.iloc[i_cal].copy()
    entrenamiento = p13b[~p13b[ID].isin(ids_13j)].copy()
    pruebas = {
        "2014B": p14b[
            ~p14b[ID].isin(ids_13b.union(ids_13j))
        ].copy(),
        "2014J": p14j[
            ~p14j[ID].isin(ids_13b.union(ids_13j).union(ids_14b))
        ].copy()
    }

    conjuntos = {
        "entrenamiento": set(entrenamiento[ID]),
        "calibracion": set(calibracion[ID]),
        "prueba_2014B": set(pruebas["2014B"][ID]),
        "prueba_2014J": set(pruebas["2014J"][ID])
    }
    nombres = list(conjuntos)
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            if conjuntos[nombres[i]] & conjuntos[nombres[j]]:
                raise AssertionError("Hay estudiantes compartidos entre cohortes.")

    modelos = {
        "estudiante_solo": VARIABLES_ESTUDIANTE,
        "contexto_curso_solo": VARIABLES_CONTEXTO,
        "combinado_sin_identidad_modulo":
            VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
    }
    filas = []

    for nombre, variables in modelos.items():
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
        prob_cal = modelo.predict_proba(x_cal)[:, 1]
        calibrador = LogisticRegression(max_iter=1000)
        calibrador.fit(
            prob_cal.reshape(-1, 1),
            calibracion[OBJETIVO].astype(int)
        )

        for periodo, datos_originales in pruebas.items():
            datos = datos_originales.reset_index(drop=True)
            x_test = preparar(datos, variables).reindex(
                columns=x_train.columns, fill_value=0
            )
            prob_base = modelo.predict_proba(x_test)[:, 1]
            prob = calibrador.predict_proba(
                prob_base.reshape(-1, 1)
            )[:, 1]
            y = datos[OBJETIVO].astype(int).to_numpy()

            for capacidad in CAPACIDADES:
                pred_fijo, corte, n_empate, plazas = seleccionar_top_k(
                    prob, capacidad, rng=None
                )
                fijo = metricas(y, pred_fijo)
                muestras = {"F1": [], "precision": [], "recall": []}
                rng = np.random.default_rng(
                    SEMILLA + len(filas) * 100 + int(capacidad * 100)
                )

                for _ in range(REPETICIONES):
                    pred = np.zeros(len(y), dtype=int)
                    k = int(np.floor(len(y) * capacidad))
                    mayores = np.flatnonzero(prob > corte)
                    empatados = np.flatnonzero(prob == corte)
                    pred[mayores] = 1
                    if plazas > 0:
                        elegidos = rng.choice(
                            empatados, size=plazas, replace=False
                        )
                        pred[elegidos] = 1
                    m = metricas(y, pred)
                    for metrica in muestras:
                        muestras[metrica].append(m[metrica])

                for metrica in ("F1", "precision", "recall"):
                    media, inferior, superior = resumen(muestras[metrica])
                    filas.append({
                        "modelo": nombre,
                        "periodo": periodo,
                        "capacidad": capacidad,
                        "metrica": metrica,
                        "valor_desempate_orden_fila": fijo[metrica],
                        "media_desempate_aleatorio": media,
                        "IC95_simulacion_inferior": inferior,
                        "IC95_simulacion_superior": superior,
                        "diferencia_media_menos_orden_fila":
                            media - fijo[metrica],
                        "matriculas": len(y),
                        "alertas_top_k": int(np.floor(len(y) * capacidad)),
                        "filas_empatadas_en_corte": n_empate,
                        "plazas_asignadas_aleatoriamente": plazas,
                        "repeticiones": REPETICIONES
                    })

    resultados = pd.DataFrame(filas)
    ruta = SALIDA / "159_sensibilidad_desempate_aleatorio.csv"
    resultados.to_csv(ruta, index=False)

    print("\nEXPERIMENTO 159: SENSIBILIDAD AL DESEMPATE ALEATORIO")
    print("Repeticiones por modelo, periodo y capacidad:", REPETICIONES)
    print("Los intervalos describen variacion por desempate, no IC poblacionales.")
    print("\nRESULTADOS")
    print(resultados.round(5).to_string(index=False))
    print("\nARCHIVO GUARDADO:")
    print(ruta)


if __name__ == "__main__":
    main()
