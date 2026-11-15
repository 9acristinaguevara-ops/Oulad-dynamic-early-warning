import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)

# =====================================================
# 1. RUTAS Y CONFIGURACION
# =====================================================

BASE = Path(__file__).resolve().parent.parent

ARCHIVO = (
    BASE / "data" / "interim" /
    "week4_combined_features_v4.parquet"
)

SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVO_RESULTADOS = (
    SALIDA / "143_validacion_separada_por_estudiante.csv"
)

ARCHIVO_PARTICIONES = (
    SALIDA / "143_comprobacion_particiones_2013J.csv"
)

VARIABLES = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "clics_recientes_3_4",
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34"
]

OBJETIVO = "target_future_withdrawal"
ID_ESTUDIANTE = "id_student"

# =====================================================
# 2. CARGAR Y SEPARAR LOS PERIODOS
# =====================================================

df = pd.read_parquet(ARCHIVO)

columnas_necesarias = (
    VARIABLES + [OBJETIVO, ID_ESTUDIANTE, "code_presentation"]
)

faltantes = [
    columna for columna in columnas_necesarias
    if columna not in df.columns
]

if faltantes:
    raise ValueError(
        "Faltan columnas necesarias: " + ", ".join(faltantes)
    )

if df[columnas_necesarias].isnull().any().any():
    # Los predictores pueden contener valores ausentes, pero no
    # deben faltar el identificador, el objetivo o la presentación.
    columnas_criticas = [
        OBJETIVO, ID_ESTUDIANTE, "code_presentation"
    ]
    if df[columnas_criticas].isnull().any().any():
        raise ValueError(
            "Hay valores ausentes en el objetivo, el ID o la presentación."
        )

train = df[df["code_presentation"] == "2013B"].copy()
validacion = df[df["code_presentation"] == "2013J"].copy()
prueba_b = df[df["code_presentation"] == "2014B"].copy()
prueba_j = df[df["code_presentation"] == "2014J"].copy()

if any(
    periodo.empty
    for periodo in [train, validacion, prueba_b, prueba_j]
):
    raise ValueError("Uno de los periodos no tiene datos.")

# =====================================================
# 3. DIVIDIR 2013J AGRUPANDO POR ESTUDIANTE
# =====================================================

# Cada estudiante queda completamente en calibracion o en
# seleccion del umbral, aunque tenga varias matriculas.
divisor = StratifiedGroupKFold(
    n_splits=2,
    shuffle=True,
    random_state=42
)

indices_cal, indices_umbral = next(
    divisor.split(
        validacion[VARIABLES],
        validacion[OBJETIVO],
        groups=validacion[ID_ESTUDIANTE]
    )
)

calibracion = validacion.iloc[indices_cal].copy()
seleccion_umbral = validacion.iloc[indices_umbral].copy()

ids_calibracion = set(calibracion[ID_ESTUDIANTE])
ids_umbral = set(seleccion_umbral[ID_ESTUDIANTE])
ids_compartidos = ids_calibracion.intersection(ids_umbral)

if ids_compartidos:
    raise AssertionError(
        "La particion por estudiante fallo: hay IDs compartidos."
    )

if calibracion[OBJETIVO].nunique() < 2:
    raise ValueError(
        "La particion de calibracion no contiene ambas clases."
    )

if seleccion_umbral[OBJETIVO].nunique() < 2:
    raise ValueError(
        "La particion de seleccion del umbral no contiene ambas clases."
    )

# Guardar evidencia de la asignacion de cada matricula.
particiones = pd.concat([
    calibracion.assign(particion="calibracion"),
    seleccion_umbral.assign(particion="seleccion_umbral")
], ignore_index=True)

particiones[
    ["code_module", "code_presentation", ID_ESTUDIANTE,
     OBJETIVO, "particion"]
].sort_values(
    [ID_ESTUDIANTE, "code_module"]
).to_csv(ARCHIVO_PARTICIONES, index=False)

print("\nPARTICION DE 2013J POR ESTUDIANTE")
for nombre, datos in [
    ("Calibracion", calibracion),
    ("Seleccion del umbral", seleccion_umbral)
]:
    print(
        nombre,
        "| registros:", len(datos),
        "| estudiantes unicos:", datos[ID_ESTUDIANTE].nunique(),
        "| retiros:", int(datos[OBJETIVO].sum()),
        "| tasa:", round(datos[OBJETIVO].mean(), 4)
    )

print("IDs compartidos entre particiones:", len(ids_compartidos))

# =====================================================
# 4. ENTRENAR EL MODELO BASE
# =====================================================

X_train = train[VARIABLES].fillna(0)
y_train = train[OBJETIVO].astype(int)

modelo = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

modelo.fit(X_train, y_train)

# =====================================================
# 5. CALIBRAR CON LA PARTICION DE CALIBRACION
# =====================================================

X_cal = calibracion[VARIABLES].fillna(0)
y_cal = calibracion[OBJETIVO].astype(int)

prob_cal = modelo.predict_proba(X_cal)[:, 1]

calibrador = LogisticRegression(max_iter=1000)
calibrador.fit(
    prob_cal.reshape(-1, 1),
    y_cal
)

# =====================================================
# 6. SELECCIONAR EL UMBRAL EN LA OTRA PARTICION
# =====================================================

X_umbral = seleccion_umbral[VARIABLES].fillna(0)
y_umbral = seleccion_umbral[OBJETIVO].astype(int)

prob_umbral_original = modelo.predict_proba(X_umbral)[:, 1]
prob_umbral_calibrada = calibrador.predict_proba(
    prob_umbral_original.reshape(-1, 1)
)[:, 1]

umbrales = [
    0.10, 0.12, 0.15, 0.18,
    0.20, 0.22, 0.25, 0.30
]

mejor_umbral = umbrales[0]
mejor_f1 = -1

for umbral in umbrales:
    predicciones = (
        prob_umbral_calibrada >= umbral
    ).astype(int)

    f1 = f1_score(
        y_umbral,
        predicciones,
        zero_division=0
    )

    if f1 > mejor_f1:
        mejor_f1 = f1
        mejor_umbral = umbral

print("\nSELECCION DEL UMBRAL")
print("Umbral elegido:", mejor_umbral)
print("F1 en seleccion:", round(mejor_f1, 4))

# =====================================================
# 7. EVALUAR LOS PERIODOS DE PRUEBA
# =====================================================

def evaluar(nombre, periodo, datos, probabilidades, umbral):
    y_real = datos[OBJETIVO].astype(int)

    predicciones = (
        probabilidades >= umbral
    ).astype(int)

    return {
        "modelo": nombre,
        "periodo": periodo,
        "registros": len(datos),
        "estudiantes_unicos": datos[ID_ESTUDIANTE].nunique(),
        "retiros_reales": int(y_real.sum()),
        "umbral": umbral,
        "AUC": roc_auc_score(y_real, probabilidades),
        "F1": f1_score(
            y_real, predicciones, zero_division=0
        ),
        "precision": precision_score(
            y_real, predicciones, zero_division=0
        ),
        "recall": recall_score(
            y_real, predicciones, zero_division=0
        ),
        "Brier": brier_score_loss(y_real, probabilidades),
        "alertas": int(predicciones.sum()),
        "porcentaje_alertas": round(
            predicciones.mean() * 100, 2
        )
    }

resultados = []

for periodo, datos in [
    ("2014B", prueba_b),
    ("2014J", prueba_j)
]:
    X = datos[VARIABLES].fillna(0)
    y = datos[OBJETIVO].astype(int)

    prob_original = modelo.predict_proba(X)[:, 1]
    prob_calibrada = calibrador.predict_proba(
        prob_original.reshape(-1, 1)
    )[:, 1]

    resultados.append(
        evaluar(
            "Logistica_original",
            periodo,
            datos,
            prob_original,
            0.45
        )
    )

    resultados.append(
        evaluar(
            "Logistica_calibrada_agrupada",
            periodo,
            datos,
            prob_calibrada,
            mejor_umbral
        )
    )

tabla = pd.DataFrame(resultados)
tabla.to_csv(ARCHIVO_RESULTADOS, index=False)

print("\nRESULTADOS DEL EXPERIMENTO 143")
print(tabla.round(4).to_string(index=False))

print("\nARCHIVOS GUARDADOS:")
print(ARCHIVO_RESULTADOS)
print(ARCHIVO_PARTICIONES)
