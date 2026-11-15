"""Experimento 158: auditoria de empates en rankings top-k.

Reproduce los modelos y cohortes independientes de 150/156/157. Cuantifica
empates en las probabilidades y, especialmente, en el punto de corte top-k.
No modifica ni reentrena con datos de prueba.
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


def preparar(datos, variables):
    x = datos[variables].copy()
    for columna in variables:
        x[columna] = pd.to_numeric(
            x[columna], errors="coerce"
        ).fillna(0)
    return x


def diagnostico_empates(datos, probabilidades, capacidad):
    n = len(datos)
    k = int(np.floor(n * capacidad))
    tabla = pd.DataFrame({
        "probabilidad": np.asarray(probabilidades),
        "posicion": np.arange(n)
    })
    tabla = tabla.sort_values(
        ["probabilidad", "posicion"],
        ascending=[False, True],
        kind="mergesort"
    ).reset_index(drop=True)

    corte = float(tabla.loc[k - 1, "probabilidad"])
    mayores = int((tabla["probabilidad"] > corte).sum())
    iguales = int((tabla["probabilidad"] == corte).sum())
    seleccionados_del_empate = k - mayores
    conteos = pd.Series(probabilidades).value_counts()

    return {
        "matriculas": n,
        "alertas_top_k": k,
        "probabilidades_distintas": int(pd.Series(probabilidades).nunique()),
        "tamano_empate_mayor": int(conteos.max()),
        "porcentaje_en_empates": float(
            (conteos[conteos > 1].sum() / n) * 100
        ),
        "probabilidad_corte": corte,
        "filas_con_probabilidad_mayor_al_corte": mayores,
        "filas_empatadas_en_corte": iguales,
        "empatados_que_entran": seleccionados_del_empate,
        "fraccion_alertas_decidida_por_desempate": (
            seleccionados_del_empate / k if k else np.nan
        )
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

    p13b = df[df[PRESENTACION] == "2013B"].copy()
    p13j = df[df[PRESENTACION] == "2013J"].copy()
    p14b = df[df[PRESENTACION] == "2014B"].copy()
    p14j = df[df[PRESENTACION] == "2014J"].copy()

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
    conjuntos = {nombre: set(d[ID]) for nombre, d in cohortes.items()}
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

        for periodo, datos in pruebas.items():
            datos = datos.reset_index(drop=True)
            x_test = preparar(datos, variables).reindex(
                columns=x_train.columns, fill_value=0
            )
            prob_base = modelo.predict_proba(x_test)[:, 1]
            prob = calibrador.predict_proba(
                prob_base.reshape(-1, 1)
            )[:, 1]

            for capacidad in CAPACIDADES:
                resultado = diagnostico_empates(
                    datos, prob, capacidad
                )
                resultado.update({
                    "modelo": nombre,
                    "periodo": periodo,
                    "capacidad": capacidad
                })
                filas.append(resultado)

    resultados = pd.DataFrame(filas)
    columnas = [
        "modelo", "periodo", "capacidad", "matriculas",
        "alertas_top_k", "probabilidades_distintas",
        "tamano_empate_mayor", "porcentaje_en_empates",
        "probabilidad_corte", "filas_con_probabilidad_mayor_al_corte",
        "filas_empatadas_en_corte", "empatados_que_entran",
        "fraccion_alertas_decidida_por_desempate"
    ]
    resultados = resultados[columnas]
    ruta = SALIDA / "158_auditoria_empates_top_k.csv"
    resultados.to_csv(ruta, index=False)

    print("\nEXPERIMENTO 158: AUDITORIA DE EMPATES TOP-K")
    print("Se examinan los empates en la probabilidad de corte.")
    print("La fraccion de alertas decidida por desempate mide cuantas")
    print("plazas top-k dependen del criterio secundario de ordenamiento.")
    print("\nRESULTADOS")
    print(resultados.round(5).to_string(index=False))
    print("\nARCHIVO GUARDADO:")
    print(ruta)


if __name__ == "__main__":
    main()
