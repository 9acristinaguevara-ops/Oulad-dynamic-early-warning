import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

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
MODULO = "code_module"

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
UMBRALES = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

# =====================================================
# 2. CARGAR Y RECONSTRUIR COHORTES
# =====================================================

df = pd.read_parquet(ARCHIVO)

necesarias = (
    VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO
    + [OBJETIVO, ID, PRESENTACION, MODULO]
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

# =====================================================
# 3. AUDITORIA DE MODULOS Y PREDICTORES
# =====================================================

modulos_train = set(entrenamiento[MODULO].astype(str).unique())
filas_modulos = []

for nombre, datos in cohortes.items():
    conteos = datos[MODULO].astype(str).value_counts()
    for modulo, cantidad in conteos.items():
        filas_modulos.append({
            "cohorte": nombre,
            "modulo": modulo,
            "matriculas": int(cantidad),
            "estudiantes": datos.loc[
                datos[MODULO].astype(str) == modulo, ID
            ].nunique(),
            "modulo_visto_en_entrenamiento": modulo in modulos_train
        })

tabla_modulos = pd.DataFrame(filas_modulos)
tabla_modulos.to_csv(
    SALIDA / "149_distribucion_modulos.csv", index=False
)

filas_resumen = []
for nombre, datos in cohortes.items():
    modulos = set(datos[MODULO].astype(str).unique())
    desconocidos = modulos - modulos_train
    mascara_desconocidos = datos[MODULO].astype(str).isin(desconocidos)

    fila = {
        "cohorte": nombre,
        "matriculas": len(datos),
        "estudiantes": datos[ID].nunique(),
        "modulos_presentes": len(modulos),
        "modulos_no_vistos": ";".join(sorted(desconocidos)),
        "matriculas_modulo_no_visto": int(mascara_desconocidos.sum()),
        "porcentaje_modulo_no_visto": mascara_desconocidos.mean() * 100
    }

    for variable in VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO:
        valores = pd.to_numeric(datos[variable], errors="coerce")
        fila[variable + "_faltantes"] = int(valores.isna().sum())

    filas_resumen.append(fila)

tabla_resumen = pd.DataFrame(filas_resumen)
tabla_resumen.to_csv(
    SALIDA / "149_resumen_modulos_y_faltantes.csv", index=False
)

# Comprobar si evaluaciones programadas varía entre módulos/presentaciones.
grupos_curso = df.groupby(
    [MODULO, PRESENTACION], dropna=False
)["evaluaciones_programadas_34"].agg(
    ["count", "nunique", "min", "max"]
).reset_index()
grupos_curso.to_csv(
    SALIDA / "149_variacion_contexto_curso.csv", index=False
)

# =====================================================
# 4. REPRODUCIR MODELOS Y DIAGNOSTICAR PROBABILIDADES
# =====================================================

def preparar(datos, numericas, incluir_modulo):
    x = datos[numericas].copy()
    for columna in numericas:
        x[columna] = pd.to_numeric(
            x[columna], errors="coerce"
        ).fillna(0)

    if incluir_modulo:
        modulo = datos[MODULO].astype(str)
        desconocido = (~modulo.isin(modulos_train)).astype(int)
        categorias = pd.Categorical(
            modulo, categories=sorted(modulos_train)
        )
        dummies = pd.get_dummies(
            categorias, prefix="modulo", dtype=int
        )
        dummies.index = datos.index
        x = pd.concat([x, dummies], axis=1)

    return x

def elegir_umbral(prob, y):
    mejor_umbral = UMBRALES[0]
    mejor_f1 = -1
    for umbral in UMBRALES:
        pred = (prob >= umbral).astype(int)
        valor = f1_score(y, pred, zero_division=0)
        if valor > mejor_f1:
            mejor_f1 = valor
            mejor_umbral = umbral
    return mejor_umbral, mejor_f1

especificaciones = {
    "estudiante_solo": (VARIABLES_ESTUDIANTE, False),
    "contexto_curso_solo": (VARIABLES_CONTEXTO, True),
    "combinado": (VARIABLES_ESTUDIANTE + VARIABLES_CONTEXTO, True)
}

filas_prob = []
filas_umbral = []

for nombre, (numericas, incluir_modulo) in especificaciones.items():
    x_train = preparar(entrenamiento, numericas, incluir_modulo)
    columnas = x_train.columns

    x_cal = preparar(calibracion, numericas, incluir_modulo).reindex(
        columns=columnas, fill_value=0
    )
    x_sel = preparar(seleccion_umbral, numericas, incluir_modulo).reindex(
        columns=columnas, fill_value=0
    )

    modelo = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
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

    umbral, f1_sel = elegir_umbral(
        prob_sel, seleccion_umbral[OBJETIVO].astype(int)
    )
    filas_umbral.append({
        "modelo": nombre,
        "predictores": len(columnas),
        "umbral_seleccionado": umbral,
        "F1_seleccion": f1_sel,
        "prob_min_seleccion": prob_sel.min(),
        "prob_mediana_seleccion": np.median(prob_sel),
        "prob_max_seleccion": prob_sel.max()
    })

    for periodo, datos in [
        ("2014B", prueba_b), ("2014J", prueba_j)
    ]:
        x_test = preparar(datos, numericas, incluir_modulo).reindex(
            columns=columnas, fill_value=0
        )
        prob_base = modelo.predict_proba(x_test)[:, 1]
        prob = calibrador.predict_proba(
            prob_base.reshape(-1, 1)
        )[:, 1]

        y = datos[OBJETIVO].astype(int).to_numpy()

        fila = {
            "modelo": nombre,
            "periodo": periodo,
            "AUC": roc_auc_score(y, prob),
            "umbral_seleccionado": umbral,
            "prob_min": prob.min(),
            "prob_p10": np.quantile(prob, 0.10),
            "prob_p25": np.quantile(prob, 0.25),
            "prob_mediana": np.median(prob),
            "prob_p75": np.quantile(prob, 0.75),
            "prob_p90": np.quantile(prob, 0.90),
            "prob_max": prob.max()
        }

        for corte in UMBRALES:
            fila["alertas_umbral_" + str(corte)] = int(
                (prob >= corte).sum()
            )
            fila["porcentaje_umbral_" + str(corte)] = (
                (prob >= corte).mean() * 100
            )

        filas_prob.append(fila)

tabla_prob = pd.DataFrame(filas_prob)
tabla_umbrales = pd.DataFrame(filas_umbral)

tabla_prob.to_csv(
    SALIDA / "149_distribucion_probabilidades_y_alertas.csv",
    index=False
)
tabla_umbrales.to_csv(
    SALIDA / "149_diagnostico_umbrales_seleccion.csv",
    index=False
)

# =====================================================
# 5. INFORME
# =====================================================

print("\nAUDITORIA DE MODULOS POR COHORTE")
print(tabla_resumen.round(4).to_string(index=False))

print("\nDISTRIBUCION DE MODULOS")
print(tabla_modulos.to_string(index=False))

print("\nVARIACION DE EVALUACIONES PROGRAMADAS POR CURSO")
print(grupos_curso.round(3).to_string(index=False))

print("\nUMBRALES ELEGIDOS EN SELECCION")
print(tabla_umbrales.round(4).to_string(index=False))

print("\nDISTRIBUCION DE PROBABILIDADES Y ALERTAS")
print(tabla_prob.round(4).to_string(index=False))

print("\nARCHIVOS GUARDADOS EN:", SALIDA)
