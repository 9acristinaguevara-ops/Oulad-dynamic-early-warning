"""Experimento 165: bootstrap de entrenamiento y calibracion para top-k.

En cada replica se remuestrean estudiantes completos, por separado, en las
cohortes de entrenamiento y calibracion. Se vuelven a ajustar los tres modelos
y sus calibradores. Las cohortes de prueba permanecen fijas; dentro de cada
replica se recalcula el ranking top-k sobre toda la cohorte de prueba.

Por tanto, los intervalos cuantifican incertidumbre debida al entrenamiento,
la calibracion y la variacion de la muestra de estudiantes de entrenamiento/
calibracion. No incorporan incertidumbre de muestreo de la cohorte de prueba.
Las 12 comparaciones principales de F1 usan intervalos Bonferroni.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score

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
MODELOS = {
    "estudiante_solo": V_EST,
    "contexto_curso_solo": V_CONT,
    "combinado_sin_identidad_modulo": V_EST + V_CONT
}
CAPACIDADES = [0.20, 0.30, 0.40]
PARES = [
    ("estudiante_solo", "combinado_sin_identidad_modulo"),
    ("contexto_curso_solo", "combinado_sin_identidad_modulo")
]
REPLICAS = 1000
SEMILLA = 165
ALPHA_FAMILIA = 0.05
N_COMPARACIONES = 12
ALPHA_AJUSTADO = ALPHA_FAMILIA / N_COMPARACIONES


def preparar(datos, variables):
    x = datos[variables].copy()
    for columna in variables:
        x[columna] = pd.to_numeric(
            x[columna], errors="coerce"
        ).fillna(0)
    return x


def remuestrear_estudiantes(datos, rng):
    """Muestrea estudiantes con reemplazo y conserva todas sus matriculas."""
    ids = datos[ID].drop_duplicates().to_numpy()
    muestra_ids = rng.choice(ids, size=len(ids), replace=True)
    indices_por_id = {
        estudiante: np.flatnonzero(datos[ID].to_numpy() == estudiante)
        for estudiante in ids
    }
    indices = []
    for estudiante in muestra_ids:
        indices.extend(indices_por_id[estudiante].tolist())
    muestra = datos.iloc[indices].copy().reset_index(drop=True)
    if muestra[TARGET].nunique() != 2:
        raise RuntimeError(
            "Una replica bootstrap no contiene ambas clases. "
            "Aumente el numero de replicas o revise la cohorte."
        )
    return muestra


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
    alerta = np.zeros(n, dtype=np.int8)
    alerta[orden.iloc[:k]["_pos"].to_numpy(dtype=int)] = 1
    return alerta


def ajustar_modelo_y_calibrador(entrenamiento, calibracion, variables):
    x_train = preparar(entrenamiento, variables)
    y_train = entrenamiento[TARGET].astype(int)

    modelo = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
    modelo.fit(x_train, y_train)

    x_cal = preparar(calibracion, variables).reindex(
        columns=x_train.columns, fill_value=0
    )
    p_cal_base = modelo.predict_proba(x_cal)[:, 1]
    calibrador = LogisticRegression(max_iter=1000)
    calibrador.fit(
        p_cal_base.reshape(-1, 1),
        calibracion[TARGET].astype(int)
    )
    return modelo, calibrador, x_train.columns


def metricas(y, alerta):
    return {
        "precision": precision_score(y, alerta, zero_division=0),
        "recall": recall_score(y, alerta, zero_division=0),
        "f1": f1_score(y, alerta, zero_division=0)
    }


def resumen(valores):
    a = np.asarray(valores, dtype=float)
    a = a[np.isfinite(a)]
    return (
        float(np.mean(a)),
        float(np.quantile(a, ALPHA_AJUSTADO / 2)),
        float(np.quantile(a, 1 - ALPHA_AJUSTADO / 2)),
        float(np.mean(a > 0)),
        len(a)
    )


def construir_cohortes(df):
    p13b = df[df[PRES] == "2013B"].copy()
    p13j = df[df[PRES] == "2013J"].copy()
    p14b = df[df[PRES] == "2014B"].copy()
    p14j = df[df[PRES] == "2014J"].copy()

    ids13b = set(p13b[ID])
    ids13j = set(p13j[ID])
    ids14b = set(p14b[ID])

    divisor = StratifiedGroupKFold(
        n_splits=2, shuffle=True, random_state=42
    )
    i_cal, _ = next(
        divisor.split(p13j, p13j[TARGET], groups=p13j[ID])
    )
    calibracion = p13j.iloc[i_cal].copy()
    entrenamiento = p13b[~p13b[ID].isin(ids13j)].copy()
    pruebas = {
        "2014B": p14b[
            ~p14b[ID].isin(ids13b.union(ids13j))
        ].copy().reset_index(drop=True),
        "2014J": p14j[
            ~p14j[ID].isin(ids13b.union(ids13j).union(ids14b))
        ].copy().reset_index(drop=True)
    }

    cohortes = [entrenamiento, calibracion, pruebas["2014B"], pruebas["2014J"]]
    for i in range(len(cohortes)):
        for j in range(i + 1, len(cohortes)):
            if set(cohortes[i][ID]) & set(cohortes[j][ID]):
                raise AssertionError(
                    "Hay estudiantes compartidos entre cohortes."
                )
    for nombre, datos in [
        ("entrenamiento", entrenamiento),
        ("calibracion", calibracion),
        ("prueba 2014B", pruebas["2014B"]),
        ("prueba 2014J", pruebas["2014J"])
    ]:
        if datos.empty or datos[TARGET].nunique() != 2:
            raise ValueError("Cohorte invalida: " + nombre)

    return entrenamiento, calibracion, pruebas


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rng = np.random.default_rng(SEMILLA)

    df = pd.read_parquet(ARCHIVO)
    necesarias = (
        V_EST + V_CONT + [TARGET, ID, PRES] + CLAVE[:1]
    )
    faltantes = [c for c in necesarias if c not in df.columns]
    if faltantes:
        raise ValueError("Faltan columnas: " + ", ".join(faltantes))

    entrenamiento, calibracion, pruebas = construir_cohortes(df)
    y_prueba = {
        periodo: datos[TARGET].astype(int).to_numpy()
        for periodo, datos in pruebas.items()
    }

    print("EXPERIMENTO 165: BOOTSTRAP DE REENTRENAMIENTO Y CALIBRACION")
    print("Replicas:", REPLICAS)
    print("Modelos reentrenados y recalibrados en cada replica:", len(MODELOS))
    print("Cohortes de prueba fijas; ranking top-k recalculado en cada replica.")
    print("\nCOHORTES")
    for nombre, datos in [
        ("entrenamiento", entrenamiento),
        ("calibracion", calibracion),
        ("prueba_2014B", pruebas["2014B"]),
        ("prueba_2014J", pruebas["2014J"])
    ]:
        print(
            nombre, "| matriculas:", len(datos),
            "| estudiantes:", datos[ID].nunique(),
            "| retiros:", int(datos[TARGET].sum())
        )

    diferencias = {}
    valores_modelo = {}
    for periodo in pruebas:
        for capacidad in CAPACIDADES:
            for modelo_a, modelo_b in PARES:
                diferencias[(periodo, capacidad, modelo_a, modelo_b)] = {
                    "precision": [], "recall": [], "f1": []
                }
            for nombre in MODELOS:
                valores_modelo[(periodo, capacidad, nombre)] = {
                    "precision": [], "recall": [], "f1": []
                }

    replicas_validas = 0
    for replica in range(REPLICAS):
        muestra_train = remuestrear_estudiantes(entrenamiento, rng)
        muestra_cal = remuestrear_estudiantes(calibracion, rng)

        predicciones = {}
        for nombre, variables in MODELOS.items():
            modelo, calibrador, columnas = ajustar_modelo_y_calibrador(
                muestra_train, muestra_cal, variables
            )
            for periodo, datos_test in pruebas.items():
                x_test = preparar(datos_test, variables).reindex(
                    columns=columnas, fill_value=0
                )
                p_base = modelo.predict_proba(x_test)[:, 1]
                predicciones[(nombre, periodo)] = calibrador.predict_proba(
                    p_base.reshape(-1, 1)
                )[:, 1]

        for periodo, datos_test in pruebas.items():
            y = y_prueba[periodo]
            alertas_por_modelo = {}
            for nombre in MODELOS:
                for capacidad in CAPACIDADES:
                    alerta = seleccionar_top_k(
                        datos_test,
                        predicciones[(nombre, periodo)],
                        capacidad
                    )
                    alertas_por_modelo[(nombre, capacidad)] = alerta
                    m = metricas(y, alerta)
                    for metrica, valor in m.items():
                        valores_modelo[(periodo, capacidad, nombre)][metrica].append(valor)

            for capacidad in CAPACIDADES:
                for modelo_a, modelo_b in PARES:
                    ma = metricas(
                        y, alertas_por_modelo[(modelo_a, capacidad)]
                    )
                    mb = metricas(
                        y, alertas_por_modelo[(modelo_b, capacidad)]
                    )
                    clave = (periodo, capacidad, modelo_a, modelo_b)
                    for metrica in ["precision", "recall", "f1"]:
                        diferencias[clave][metrica].append(
                            mb[metrica] - ma[metrica]
                        )

        replicas_validas += 1
        if (replica + 1) % 100 == 0:
            print("Replicas completadas:", replica + 1, "/", REPLICAS)

    filas = []
    for clave, distribuciones in diferencias.items():
        periodo, capacidad, modelo_a, modelo_b = clave
        fila = {
            "periodo": periodo,
            "capacidad": capacidad,
            "modelo_a": modelo_a,
            "modelo_b": modelo_b,
            "replicas_validas": replicas_validas,
            "matriculas_prueba": len(pruebas[periodo]),
            "estudiantes_prueba": pruebas[periodo][ID].nunique()
        }
        for metrica, valores in distribuciones.items():
            media, inferior, superior, prop_positiva, n = resumen(valores)
            fila["media_diferencia_" + metrica] = media
            fila["ic_simultaneo_inferior_" + metrica] = inferior
            fila["ic_simultaneo_superior_" + metrica] = superior
            fila["proporcion_replicas_diferencia_positiva_" + metrica] = prop_positiva
            fila["replicas_validas_" + metrica] = n

        dist_f1 = distribuciones["f1"]
        fila["f1_diferencia_observada_referencia"] = float(np.mean(dist_f1))
        filas.append(fila)

    resultados = pd.DataFrame(filas)
    ruta = SALIDA / "165_bootstrap_reentrenamiento_calibracion_top_k.csv"
    resultados.to_csv(ruta, index=False)

    filas_modelos = []
    for clave, distribuciones in valores_modelo.items():
        periodo, capacidad, nombre = clave
        fila = {
            "periodo": periodo,
            "capacidad": capacidad,
            "modelo": nombre
        }
        for metrica, valores in distribuciones.items():
            fila["media_bootstrap_" + metrica] = float(np.mean(valores))
            fila["p025_bootstrap_" + metrica] = float(np.quantile(valores, 0.025))
            fila["p975_bootstrap_" + metrica] = float(np.quantile(valores, 0.975))
        filas_modelos.append(fila)

    tabla_modelos = pd.DataFrame(filas_modelos)
    ruta_modelos = SALIDA / "165_distribucion_metricas_modelos.csv"
    tabla_modelos.to_csv(ruta_modelos, index=False)

    print("\nDIFERENCIAS ENTRE MODELOS")
    print(resultados.round(5).to_string(index=False))
    print("\nDISTRIBUCION DE METRICAS POR MODELO")
    print(tabla_modelos.round(5).to_string(index=False))
    print("\nARCHIVOS GUARDADOS:")
    print(ruta)
    print(ruta_modelos)
    print(
        "\nNota: intervalos Bonferroni para las 12 comparaciones principales "
        "de F1. Se reentrenan y recalibran modelos; las cohortes de prueba "
        "permanecen fijas. No se incorpora incertidumbre de muestreo de prueba."
    )


if __name__ == "__main__":
    main()
