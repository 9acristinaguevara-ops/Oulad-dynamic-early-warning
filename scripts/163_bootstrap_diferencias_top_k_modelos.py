"""Experimento 163: bootstrap pareado por estudiante de diferencias top-k.

Reproduce la cohorte/modelos del experimento 162. Compara precision, recall y F1
entre pares de modelos, con selección top-k determinista idéntica.
El bootstrap remuestrea estudiantes completos y mantiene juntas sus matrículas.
Las predicciones y las listas top-k permanecen fijas: los intervalos son
condicionales al modelo y a la regla de selección ya ajustados.
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
REPLICAS = 2000
SEMILLA = 163
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


def top_k(datos, probabilidades, capacidad):
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
    alerta = np.zeros(n, dtype=np.int8)
    alerta[elegidos] = 1
    return alerta


def metricas_ponderadas(y, alerta, pesos):
    tp = float(np.sum(pesos * y * alerta))
    fp = float(np.sum(pesos * (1 - y) * alerta))
    fn = float(np.sum(pesos * y * (1 - alerta)))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else np.nan
    )
    return precision, recall, f1


def resumen_bootstrap(diferencias):
    a = np.asarray(diferencias, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    return (
        float(np.mean(a)),
        float(np.quantile(a, 0.025)),
        float(np.quantile(a, 0.975)),
        float(np.mean(a > 0)),
        len(a)
    )


def main():
    rng = np.random.default_rng(SEMILLA)
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
    print("EXPERIMENTO 163: BOOTSTRAP PAREADO POR ESTUDIANTE")
    print("Replicas:", REPLICAS)
    print("Las predicciones y alertas top-k se mantienen fijas.")

    for periodo, datos0 in pruebas.items():
        datos = datos0.reset_index(drop=True)
        y = datos[TARGET].astype(int).to_numpy()
        ids = datos[ID].to_numpy()
        estudiantes = np.unique(ids)
        indices_por_estudiante = {
            estudiante: np.flatnonzero(ids == estudiante)
            for estudiante in estudiantes
        }

        for capacidad in CAPACIDADES:
            alertas = {
                nombre: top_k(datos, probabilidades[(nombre, periodo)], capacidad)
                for nombre in especificaciones
            }
            for modelo_a, modelo_b in PARES:
                diferencias = {"precision": [], "recall": [], "f1": []}
                # Cada replica selecciona estudiantes, no filas individuales.
                for _ in range(REPLICAS):
                    muestra = rng.choice(estudiantes, size=len(estudiantes), replace=True)
                    pesos = np.zeros(len(datos), dtype=float)
                    unicos, cantidades = np.unique(muestra, return_counts=True)
                    for estudiante, cantidad in zip(unicos, cantidades):
                        pesos[indices_por_estudiante[estudiante]] = cantidad

                    ma = metricas_ponderadas(y, alertas[modelo_a], pesos)
                    mb = metricas_ponderadas(y, alertas[modelo_b], pesos)
                    for nombre_metrica, valor_a, valor_b in zip(
                        ["precision", "recall", "f1"], ma, mb
                    ):
                        diferencias[nombre_metrica].append(valor_b - valor_a)

                metricas_a = metricas_ponderadas(
                    y, alertas[modelo_a], np.ones(len(datos))
                )
                metricas_b = metricas_ponderadas(
                    y, alertas[modelo_b], np.ones(len(datos))
                )
                fila = {
                    "periodo": periodo,
                    "capacidad": capacidad,
                    "modelo_a": modelo_a,
                    "modelo_b": modelo_b,
                    "matriculas": len(datos),
                    "estudiantes_unicos": len(estudiantes),
                    "alertas_por_modelo": int(alertas[modelo_a].sum())
                }
                for nombre_metrica, valor_a, valor_b in zip(
                    ["precision", "recall", "f1"], metricas_a, metricas_b
                ):
                    media, inferior, superior, prob_mayor_cero, n_validos = (
                        resumen_bootstrap(diferencias[nombre_metrica])
                    )
                    fila[f"{nombre_metrica}_a"] = valor_a
                    fila[f"{nombre_metrica}_b"] = valor_b
                    fila[f"diferencia_{nombre_metrica}_b_menos_a"] = valor_b - valor_a
                    fila[f"ic95_inferior_{nombre_metrica}"] = inferior
                    fila[f"ic95_superior_{nombre_metrica}"] = superior
                    fila[f"probabilidad_bootstrap_diferencia_positiva_{nombre_metrica}"] = prob_mayor_cero
                filas.append(fila)
                print(
                    periodo, capacidad, modelo_a, "vs", modelo_b,
                    "| delta F1:", round(fila["diferencia_f1_b_menos_a"], 5),
                    "| IC95%:",
                    round(fila["ic95_inferior_f1"], 5),
                    round(fila["ic95_superior_f1"], 5)
                )

    resultados = pd.DataFrame(filas)
    ruta = SALIDA / "163_bootstrap_diferencias_top_k_modelos.csv"
    resultados.to_csv(ruta, index=False)
    print("\nRESULTADOS COMPLETOS")
    print(resultados.round(5).to_string(index=False))
    print("\nARCHIVO GUARDADO:")
    print(ruta)
    print("\nNota: intervalos condicionales a los modelos, calibradores y listas top-k fijas.")


if __name__ == "__main__":
    main()
