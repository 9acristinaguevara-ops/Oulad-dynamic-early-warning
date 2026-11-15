"""Experimento 162: concordancia entre listas de alertas de los modelos.

Reproduce cohortes independientes, modelos y calibracion de los experimentos 150-161.
Compara las listas top-k de los tres modelos en cada periodo/capacidad:
- interseccion, Jaccard y solapamiento respecto a cada lista;
- retiros futuros capturados por ambos, solo por A, solo por B;
- concordancia de las alertas, sin interpretar el solapamiento como exactitud.
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

TARGET = "target_future_withdrawal"
ID = "id_student"
PRES = "code_presentation"
CLAVE = ["code_module", "code_presentation", ID]
V_EST = [
    "week3_clicks", "week4_clicks", "week4_active_days",
    "week4_activity_records", "has_week4_activity",
    "weekly_change", "clics_recientes_3_4",
    "evaluaciones_entregadas_34"
]
V_CONT = ["evaluaciones_programadas_34"]
CAPACIDADES = [0.20, 0.30, 0.40]
PARES = [
    ("estudiante_solo", "contexto_curso_solo"),
    ("estudiante_solo", "combinado_sin_identidad_modulo"),
    ("contexto_curso_solo", "combinado_sin_identidad_modulo")
]


def preparar(datos, variables):
    x = datos[variables].copy()
    for c in variables:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    return x


def seleccionar_top_k(datos, probabilidades, capacidad):
    n = len(datos)
    k = int(np.floor(n * capacidad))
    orden = datos[CLAVE].copy().reset_index(drop=True)
    orden["_prob"] = probabilidades
    orden["_pos"] = np.arange(n)
    orden = orden.sort_values(
        ["_prob", "code_module", "code_presentation", ID, "_pos"],
        ascending=[False, True, True, True, True],
        kind="mergesort"
    )
    elegidos = orden.iloc[:k]["_pos"].to_numpy(dtype=int)
    indicador = np.zeros(n, dtype=np.int8)
    indicador[elegidos] = 1
    return indicador


def main():
    df = pd.read_parquet(ARCHIVO)
    p13b = df[df[PRES] == "2013B"].copy()
    p13j = df[df[PRES] == "2013J"].copy()
    p14b = df[df[PRES] == "2014B"].copy()
    p14j = df[df[PRES] == "2014J"].copy()

    ids13b = set(p13b[ID])
    ids13j = set(p13j[ID])
    ids14b = set(p14b[ID])
    divisor = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    i_cal, _ = next(divisor.split(p13j, p13j[TARGET], groups=p13j[ID]))
    calibracion = p13j.iloc[i_cal].copy()
    entrenamiento = p13b[~p13b[ID].isin(ids13j)].copy()
    pruebas = {
        "2014B": p14b[~p14b[ID].isin(ids13b.union(ids13j))].copy(),
        "2014J": p14j[
            ~p14j[ID].isin(ids13b.union(ids13j).union(ids14b))
        ].copy()
    }

    grupos = [entrenamiento, calibracion, pruebas["2014B"], pruebas["2014J"]]
    for i in range(len(grupos)):
        for j in range(i + 1, len(grupos)):
            if set(grupos[i][ID]) & set(grupos[j][ID]):
                raise AssertionError("Se detectaron estudiantes compartidos entre cohortes.")

    especificaciones = {
        "estudiante_solo": V_EST,
        "contexto_curso_solo": V_CONT,
        "combinado_sin_identidad_modulo": V_EST + V_CONT
    }
    probabilidades = {}
    for nombre, variables in especificaciones.items():
        x_train = preparar(entrenamiento, variables)
        modelo = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        )
        modelo.fit(x_train, entrenamiento[TARGET].astype(int))

        x_cal = preparar(calibracion, variables).reindex(columns=x_train.columns, fill_value=0)
        p_cal = modelo.predict_proba(x_cal)[:, 1]
        calibrador = LogisticRegression(max_iter=1000)
        calibrador.fit(p_cal.reshape(-1, 1), calibracion[TARGET].astype(int))

        for periodo, datos0 in pruebas.items():
            datos = datos0.reset_index(drop=True)
            x_test = preparar(datos, variables).reindex(columns=x_train.columns, fill_value=0)
            p_test = modelo.predict_proba(x_test)[:, 1]
            probabilidades[(nombre, periodo)] = calibrador.predict_proba(
                p_test.reshape(-1, 1)
            )[:, 1]

    filas = []
    for periodo, datos0 in pruebas.items():
        datos = datos0.reset_index(drop=True)
        y = datos[TARGET].astype(int).to_numpy()
        for capacidad in CAPACIDADES:
            selecciones = {}
            for nombre in especificaciones:
                selecciones[nombre] = seleccionar_top_k(
                    datos, probabilidades[(nombre, periodo)], capacidad
                ).astype(bool)

            for modelo_a, modelo_b in PARES:
                a = selecciones[modelo_a]
                b = selecciones[modelo_b]
                inter = a & b
                union = a | b
                n_a, n_b = int(a.sum()), int(b.sum())
                n_inter = int(inter.sum())
                n_union = int(union.sum())
                retiros_a = int((a & (y == 1)).sum())
                retiros_b = int((b & (y == 1)).sum())
                filas.append({
                    "periodo": periodo,
                    "capacidad": capacidad,
                    "modelo_a": modelo_a,
                    "modelo_b": modelo_b,
                    "matriculas": len(datos),
                    "alertas_a": n_a,
                    "alertas_b": n_b,
                    "alertas_compartidas": n_inter,
                    "alertas_solo_a": int((a & ~b).sum()),
                    "alertas_solo_b": int((b & ~a).sum()),
                    "jaccard": n_inter / n_union if n_union else 1.0,
                    "solapamiento_sobre_a": n_inter / n_a if n_a else np.nan,
                    "solapamiento_sobre_b": n_inter / n_b if n_b else np.nan,
                    "retiros_futuros_en_a": retiros_a,
                    "retiros_futuros_en_b": retiros_b,
                    "retiros_capturados_por_ambos": int((inter & (y == 1)).sum()),
                    "retiros_solo_a": int((a & ~b & (y == 1)).sum()),
                    "retiros_solo_b": int((b & ~a & (y == 1)).sum())
                })

    resultados = pd.DataFrame(filas)
    ruta = SALIDA / "162_concordancia_listas_alertas.csv"
    resultados.to_csv(ruta, index=False)
    print("\nEXPERIMENTO 162: CONCORDANCIA ENTRE LISTAS DE ALERTAS")
    print("Se comparan los conjuntos top-k de los tres modelos.")
    print("Jaccard mide coincidencia de matrículas, no calidad predictiva.")
    print("\nRESULTADOS")
    print(resultados.round(5).to_string(index=False))
    print("\nARCHIVO GUARDADO:")
    print(ruta)


if __name__ == "__main__":
    main()
