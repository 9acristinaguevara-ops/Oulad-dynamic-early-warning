"""Experimento 161: estabilidad individual de la seleccion top-k.

Reproduce las cohortes y modelos del experimento 160. Para cada top-k:
- mide el tamaño del empate en el corte y las plazas que se sortean;
- simula 2000 selecciones aleatorias;
- calcula el solapamiento/Jaccard con la seleccion determinista oficial;
- estima la probabilidad de inclusion de cada matricula empatada.

La simulacion mantiene fijas probabilidades y etiquetas. No es un IC poblacional.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
REPETICIONES = 2000
SEMILLA = 161


def preparar(datos, variables):
    x = datos[variables].copy()
    for columna in variables:
        x[columna] = pd.to_numeric(x[columna], errors="coerce").fillna(0)
    return x


def top_k_oficial(datos, prob, capacidad):
    n = len(datos)
    k = int(np.floor(n * capacidad))
    orden = datos[CLAVE].copy().reset_index(drop=True)
    orden["_prob"] = prob
    orden["_pos"] = np.arange(n)
    orden = orden.sort_values(
        ["_prob", "code_module", "code_presentation", ID, "_pos"],
        ascending=[False, True, True, True, True],
        kind="mergesort"
    )
    corte = float(orden.iloc[k - 1]["_prob"])
    mayores = np.flatnonzero(np.asarray(prob) > corte)
    empatados = np.flatnonzero(np.asarray(prob) == corte)
    plazas = k - len(mayores)

    pred = np.zeros(n, dtype=np.int8)
    pred[mayores] = 1
    empate_df = datos.iloc[empatados][CLAVE].copy()
    empate_df["_pos"] = empatados
    empate_df = empate_df.sort_values(
        ["code_module", "code_presentation", ID, "_pos"],
        kind="mergesort"
    )
    elegidos = empate_df["_pos"].to_numpy()[:plazas]
    pred[elegidos] = 1
    return pred, corte, mayores, empatados, plazas


def main():
    df = pd.read_parquet(ARCHIVO)
    p13b = df[df[PRESENTACION] == "2013B"].copy()
    p13j = df[df[PRESENTACION] == "2013J"].copy()
    p14b = df[df[PRESENTACION] == "2014B"].copy()
    p14j = df[df[PRESENTACION] == "2014J"].copy()

    ids_13b, ids_13j = set(p13b[ID]), set(p13j[ID])
    ids_14b = set(p14b[ID])
    divisor = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    i_cal, _ = next(divisor.split(p13j, p13j[OBJETIVO], groups=p13j[ID]))
    calibracion = p13j.iloc[i_cal].copy()
    entrenamiento = p13b[~p13b[ID].isin(ids_13j)].copy()
    pruebas = {
        "2014B": p14b[~p14b[ID].isin(ids_13b.union(ids_13j))].copy(),
        "2014J": p14j[
            ~p14j[ID].isin(ids_13b.union(ids_13j).union(ids_14b))
        ].copy()
    }

    cohortes = [entrenamiento, calibracion, pruebas["2014B"], pruebas["2014J"]]
    for i in range(len(cohortes)):
        for j in range(i + 1, len(cohortes)):
            if set(cohortes[i][ID]) & set(cohortes[j][ID]):
                raise AssertionError("Hay estudiantes compartidos entre cohortes.")

    modelos = {
        "estudiante_solo": VARIABLES_ESTUDIANTE,
        "contexto_curso_solo": VARIABLES_CONTEXTO,
        "combinado_sin_identidad_modulo": VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
    }
    filas = []
    contador = 0

    for nombre, variables in modelos.items():
        x_train = preparar(entrenamiento, variables)
        modelo = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        )
        modelo.fit(x_train, entrenamiento[OBJETIVO].astype(int))

        x_cal = preparar(calibracion, variables).reindex(columns=x_train.columns, fill_value=0)
        p_cal_base = modelo.predict_proba(x_cal)[:, 1]
        calibrador = LogisticRegression(max_iter=1000)
        calibrador.fit(p_cal_base.reshape(-1, 1), calibracion[OBJETIVO].astype(int))

        for periodo, datos_originales in pruebas.items():
            datos = datos_originales.reset_index(drop=True)
            x_test = preparar(datos, variables).reindex(columns=x_train.columns, fill_value=0)
            p_base = modelo.predict_proba(x_test)[:, 1]
            prob = calibrador.predict_proba(p_base.reshape(-1, 1))[:, 1]

            for capacidad in CAPACIDADES:
                oficial, corte, mayores, empatados, plazas = top_k_oficial(
                    datos, prob, capacidad
                )
                rng = np.random.default_rng(SEMILLA + contador)
                frecuencias = np.zeros(len(empatados), dtype=np.int32)
                jaccard = []
                solapamiento = []

                oficiales = set(np.flatnonzero(oficial).tolist())
                for _ in range(REPETICIONES):
                    elegidos = rng.choice(empatados, size=plazas, replace=False) if plazas else np.array([], dtype=int)
                    seleccion = set(mayores.tolist()).union(elegidos.tolist())
                    interseccion = len(oficiales.intersection(seleccion))
                    union = len(oficiales.union(seleccion))
                    solapamiento.append(interseccion / len(oficiales))
                    jaccard.append(interseccion / union if union else 1.0)
                    if plazas:
                        posiciones = np.searchsorted(empatados, elegidos)
                        frecuencias[posiciones] += 1

                prob_inclusion = frecuencias / REPETICIONES if plazas else np.zeros(len(empatados))
                filas.append({
                    "modelo": nombre,
                    "periodo": periodo,
                    "capacidad": capacidad,
                    "matriculas": len(datos),
                    "alertas_top_k": int(np.floor(len(datos) * capacidad)),
                    "probabilidad_corte": corte,
                    "filas_con_probabilidad_mayor_al_corte": len(mayores),
                    "tamano_empate_en_corte": len(empatados),
                    "plazas_a_asignar_en_empate": plazas,
                    "probabilidad_inclusion_de_cada_empatado": plazas / len(empatados) if len(empatados) else np.nan,
                    "media_probabilidad_inclusion_simulada": float(np.mean(prob_inclusion)) if len(prob_inclusion) else np.nan,
                    "minimo_probabilidad_inclusion_simulada": float(np.min(prob_inclusion)) if len(prob_inclusion) else np.nan,
                    "maximo_probabilidad_inclusion_simulada": float(np.max(prob_inclusion)) if len(prob_inclusion) else np.nan,
                    "solapamiento_medio_con_regla_oficial": float(np.mean(solapamiento)),
                    "jaccard_medio_con_regla_oficial": float(np.mean(jaccard)),
                    "repeticiones": REPETICIONES
                })
                contador += 1

    resultados = pd.DataFrame(filas)
    ruta = SALIDA / "161_estabilidad_individual_top_k.csv"
    resultados.to_csv(ruta, index=False)
    print("\nEXPERIMENTO 161: ESTABILIDAD INDIVIDUAL DE ALERTAS TOP-K")
    print("Repeticiones por modelo, periodo y capacidad:", REPETICIONES)
    print("La probabilidad de inclusion estima la incertidumbre individual por empates.")
    print("El Jaccard compara conjuntos de alertas con la regla determinista oficial.")
    print("\nRESULTADOS")
    print(resultados.round(5).to_string(index=False))
    print("\nARCHIVO GUARDADO:")
    print(ruta)


if __name__ == "__main__":
    main()
