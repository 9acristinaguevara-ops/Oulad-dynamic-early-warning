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

ARCHIVO_RESULTADOS = SALIDA / "144_validacion_estricta_por_estudiante.csv"
ARCHIVO_COHORTES = SALIDA / "144_resumen_cohortes_independientes.csv"
ARCHIVO_SOLAPAMIENTOS = SALIDA / "144_comprobacion_solapamientos.csv"
ARCHIVO_PARTICIONES = SALIDA / "144_particiones_2013J.csv"

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
PRESENTACION = "code_presentation"

# =====================================================
# 2. CARGAR Y VALIDAR LOS DATOS
# =====================================================

df = pd.read_parquet(ARCHIVO)

columnas_necesarias = (
    VARIABLES + [OBJETIVO, ID_ESTUDIANTE, PRESENTACION, "code_module"]
)

faltantes = [
    columna for columna in columnas_necesarias
    if columna not in df.columns
]

if faltantes:
    raise ValueError(
        "Faltan columnas necesarias: " + ", ".join(faltantes)
    )

columnas_criticas = [OBJETIVO, ID_ESTUDIANTE, PRESENTACION, "code_module"]

if df[columnas_criticas].isnull().any().any():
    raise ValueError(
        "Hay valores ausentes en el objetivo, el ID o las claves del curso."
    )

if not set(df[OBJETIVO].unique()).issubset({0, 1}):
    raise ValueError("El objetivo debe contener solamente 0 y 1.")

periodos = {}

for nombre in ["2013B", "2013J", "2014B", "2014J"]:
    periodos[nombre] = df[df[PRESENTACION] == nombre].copy()

if any(datos.empty for datos in periodos.values()):
    raise ValueError("Uno de los periodos necesarios no tiene datos.")

periodo_2013B = periodos["2013B"]
periodo_2013J = periodos["2013J"]
periodo_2014B = periodos["2014B"]
periodo_2014J = periodos["2014J"]

# Identificadores completos de cada periodo, antes de excluir estudiantes.
ids_2013B = set(periodo_2013B[ID_ESTUDIANTE])
ids_2013J = set(periodo_2013J[ID_ESTUDIANTE])
ids_2014B = set(periodo_2014B[ID_ESTUDIANTE])

# =====================================================
# 3. DIVIDIR 2013J POR ESTUDIANTE
# =====================================================

divisor = StratifiedGroupKFold(
    n_splits=2,
    shuffle=True,
    random_state=42
)

indices_cal, indices_umbral = next(
    divisor.split(
        periodo_2013J[VARIABLES],
        periodo_2013J[OBJETIVO],
        groups=periodo_2013J[ID_ESTUDIANTE]
    )
)

calibracion = periodo_2013J.iloc[indices_cal].copy()
seleccion_umbral = periodo_2013J.iloc[indices_umbral].copy()

ids_calibracion = set(calibracion[ID_ESTUDIANTE])
ids_umbral = set(seleccion_umbral[ID_ESTUDIANTE])

if ids_calibracion.intersection(ids_umbral):
    raise AssertionError(
        "Hay estudiantes compartidos entre calibracion y seleccion del umbral."
    )

# =====================================================
# 4. CONSTRUIR COHORTES ESTRICTAMENTE INDEPENDIENTES
# =====================================================

# Entrenamiento: excluir del 2013B a toda persona que aparezca en 2013J.
entrenamiento = periodo_2013B[
    ~periodo_2013B[ID_ESTUDIANTE].isin(ids_2013J)
].copy()

# Prueba 2014B: estudiantes que no aparecen en 2013B ni 2013J.
ids_anteriores_2014B = ids_2013B.union(ids_2013J)

prueba_b = periodo_2014B[
    ~periodo_2014B[ID_ESTUDIANTE].isin(ids_anteriores_2014B)
].copy()

# Prueba 2014J: estudiantes que no aparecen en 2013B, 2013J ni 2014B.
ids_anteriores_2014J = ids_anteriores_2014B.union(ids_2014B)

prueba_j = periodo_2014J[
    ~periodo_2014J[ID_ESTUDIANTE].isin(ids_anteriores_2014J)
].copy()

cohortes = {
    "entrenamiento": entrenamiento,
    "calibracion": calibracion,
    "seleccion_umbral": seleccion_umbral,
    "prueba_2014B": prueba_b,
    "prueba_2014J": prueba_j
}

# Verificar que cada cohorte tenga datos y ambas clases.
for nombre, datos in cohortes.items():
    if datos.empty:
        raise ValueError("La cohorte " + nombre + " esta vacia.")

    if datos[OBJETIVO].nunique() < 2:
        raise ValueError(
            "La cohorte " + nombre + " no contiene ambas clases."
        )

# =====================================================
# 5. COMPROBAR INDEPENDENCIA ENTRE TODAS LAS COHORTES
# =====================================================

ids_cohortes = {
    nombre: set(datos[ID_ESTUDIANTE])
    for nombre, datos in cohortes.items()
}

nombres = list(ids_cohortes)
filas_solapamiento = []

for i in range(len(nombres)):
    for j in range(i + 1, len(nombres)):
        nombre_a = nombres[i]
        nombre_b = nombres[j]
        compartidos = ids_cohortes[nombre_a].intersection(
            ids_cohortes[nombre_b]
        )

        filas_solapamiento.append({
            "cohorte_a": nombre_a,
            "cohorte_b": nombre_b,
            "estudiantes_compartidos": len(compartidos)
        })

        if compartidos:
            raise AssertionError(
                "Hay estudiantes compartidos entre "
                + nombre_a + " y " + nombre_b
            )

tabla_solapamientos = pd.DataFrame(filas_solapamiento)
tabla_solapamientos.to_csv(ARCHIVO_SOLAPAMIENTOS, index=False)

# =====================================================
# 6. GUARDAR RESUMEN Y PARTICIONES
# =====================================================

filas_cohortes = []

for nombre, datos in cohortes.items():
    filas_cohortes.append({
        "cohorte": nombre,
        "matriculas": len(datos),
        "estudiantes_unicos": datos[ID_ESTUDIANTE].nunique(),
        "clase_0": int((datos[OBJETIVO] == 0).sum()),
        "clase_1": int((datos[OBJETIVO] == 1).sum()),
        "tasa_retiro": datos[OBJETIVO].mean()
    })

tabla_cohortes = pd.DataFrame(filas_cohortes)
tabla_cohortes.to_csv(ARCHIVO_COHORTES, index=False)

particiones = pd.concat([
    calibracion.assign(particion="calibracion"),
    seleccion_umbral.assign(particion="seleccion_umbral")
], ignore_index=True)

particiones[
    ["code_module", PRESENTACION, ID_ESTUDIANTE, OBJETIVO, "particion"]
].sort_values(
    [ID_ESTUDIANTE, "code_module"]
).to_csv(ARCHIVO_PARTICIONES, index=False)

# =====================================================
# 7. ENTRENAR EL MODELO BASE
# =====================================================

X_train = entrenamiento[VARIABLES].fillna(0)
y_train = entrenamiento[OBJETIVO].astype(int)

modelo = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

modelo.fit(X_train, y_train)

# =====================================================
# 8. CALIBRAR CON LA COHORTE DE CALIBRACION
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
# 9. SELECCIONAR EL UMBRAL EN SU COHORTE INDEPENDIENTE
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

# =====================================================
# 10. FUNCION DE EVALUACION
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

# =====================================================
# 11. EVALUAR SOLAMENTE ESTUDIANTES NUEVOS
# =====================================================

resultados = []

for periodo, datos in [
    ("2014B_estudiantes_nuevos", prueba_b),
    ("2014J_estudiantes_nuevos", prueba_j)
]:
    X = datos[VARIABLES].fillna(0)

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
            "Logistica_calibrada_estricta",
            periodo,
            datos,
            prob_calibrada,
            mejor_umbral
        )
    )

tabla_resultados = pd.DataFrame(resultados)
tabla_resultados.to_csv(ARCHIVO_RESULTADOS, index=False)

# =====================================================
# 12. INFORME EN CONSOLA
# =====================================================

print("\nVALIDACION ESTRICTA POR ESTUDIANTE")
print("Estudiantes compartidos entre cohortes: 0")

print("\nRESUMEN DE COHORTES")
print(tabla_cohortes.round(4).to_string(index=False))

print("\nCOMPROBACION DE SOLAPAMIENTOS")
print(tabla_solapamientos.to_string(index=False))

print("\nSELECCION DEL UMBRAL")
print("Umbral elegido:", mejor_umbral)
print("F1 en seleccion:", round(mejor_f1, 4))

print("\nRESULTADOS EN ESTUDIANTES NUEVOS")
print(tabla_resultados.round(4).to_string(index=False))

print("\nARCHIVOS GUARDADOS:")
print(ARCHIVO_RESULTADOS)
print(ARCHIVO_COHORTES)
print(ARCHIVO_SOLAPAMIENTOS)
print(ARCHIVO_PARTICIONES)
