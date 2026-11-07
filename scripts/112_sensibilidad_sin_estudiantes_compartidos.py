
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)

# =====================================================
# 1. RUTAS
# =====================================================

BASE = Path(__file__).resolve().parent.parent

archivo = (
    BASE / "data" / "interim" /
    "week4_combined_features_v4.parquet"
)

salida = BASE / "outputs" / "tables"
salida.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. CARGAR DATOS
# =====================================================

df = pd.read_parquet(archivo)

variables = [
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

objetivo = "target_future_withdrawal"


# =====================================================
# 3. SEPARAR LOS PERIODOS
# =====================================================

train = df[
    df["code_presentation"] == "2013B"
].copy()

validacion = df[
    df["code_presentation"] == "2013J"
].copy()

prueba_b = df[
    df["code_presentation"] == "2014B"
].copy()

prueba_j = df[
    df["code_presentation"] == "2014J"
].copy()

if any(
    periodo.empty
    for periodo in [train, validacion, prueba_b, prueba_j]
):
    raise ValueError("Uno de los periodos no tiene datos.")


# =====================================================
# 4. IDENTIFICAR ESTUDIANTES COMPARTIDOS
# =====================================================

estudiantes_train = set(train["id_student"])

prueba_b_original = prueba_b.copy()
prueba_j_original = prueba_j.copy()

prueba_b_sin_compartidos = prueba_b[
    ~prueba_b["id_student"].isin(estudiantes_train)
].copy()

prueba_j_sin_compartidos = prueba_j[
    ~prueba_j["id_student"].isin(estudiantes_train)
].copy()

print("\nSOLAPAMIENTO CON 2013B")

for nombre, original, filtrada in [
    ("2014B", prueba_b_original, prueba_b_sin_compartidos),
    ("2014J", prueba_j_original, prueba_j_sin_compartidos)
]:

    estudiantes_originales = original["id_student"].nunique()
    estudiantes_filtrados = filtrada["id_student"].nunique()

    print("\nPresentacion:", nombre)
    print("Registros originales:", len(original))
    print("Registros despues de excluir:", len(filtrada))
    print(
        "Registros excluidos:",
        len(original) - len(filtrada)
    )
    print("Estudiantes originales:", estudiantes_originales)
    print("Estudiantes restantes:", estudiantes_filtrados)
    print(
        "Estudiantes excluidos:",
        estudiantes_originales - estudiantes_filtrados
    )


# =====================================================
# 5. DIVIDIR 2013J EN DOS PARTES
# =====================================================

calibracion, seleccion_umbral = train_test_split(
    validacion,
    test_size=0.5,
    random_state=42,
    stratify=validacion[objetivo]
)

print("\nREGISTROS PARA CALIBRACION:", len(calibracion))
print("REGISTROS PARA SELECCION DEL UMBRAL:", len(seleccion_umbral))


# =====================================================
# 6. PREPARAR LOS DATOS
# =====================================================

X_train = train[variables].fillna(0)
y_train = train[objetivo]

X_cal = calibracion[variables].fillna(0)
y_cal = calibracion[objetivo]

X_umbral = seleccion_umbral[variables].fillna(0)
y_umbral = seleccion_umbral[objetivo]


# =====================================================
# 7. ENTRENAR EL MODELO
# =====================================================

modelo = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

modelo.fit(X_train, y_train)


# =====================================================
# 8. CALIBRAR LAS PROBABILIDADES
# =====================================================

prob_cal = modelo.predict_proba(X_cal)[:, 1]

calibrador = LogisticRegression(max_iter=1000)

calibrador.fit(
    prob_cal.reshape(-1, 1),
    y_cal
)


# =====================================================
# 9. SELECCIONAR EL UMBRAL
# =====================================================

prob_umbral_original = modelo.predict_proba(
    X_umbral
)[:, 1]

prob_umbral_cal = calibrador.predict_proba(
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
        prob_umbral_cal >= umbral
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
# 10. FUNCION DE EVALUACION
# =====================================================

def evaluar(nombre, periodo, datos, probabilidades, umbral):

    y_real = datos[objetivo]

    predicciones = (
        probabilidades >= umbral
    ).astype(int)

    return {
        "modelo": nombre,
        "periodo": periodo,
        "registros": len(datos),
        "estudiantes_unicos": datos["id_student"].nunique(),
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
        "Brier": brier_score_loss(
            y_real, probabilidades
        ),
        "alertas": int(predicciones.sum()),
        "porcentaje_alertas": round(
            predicciones.mean() * 100, 2
        )
    }


# =====================================================
# 11. EVALUAR LOS CONJUNTOS ORIGINALES Y FILTRADOS
# =====================================================

resultados = []

conjuntos = [
    ("2014B", prueba_b_original, "Original"),
    ("2014B", prueba_b_sin_compartidos, "Sin_compartidos"),
    ("2014J", prueba_j_original, "Original"),
    ("2014J", prueba_j_sin_compartidos, "Sin_compartidos")
]

for periodo, datos, tipo in conjuntos:

    X = datos[variables].fillna(0)

    y = datos[objetivo]

    # Probabilidades del modelo original
    probabilidades_originales = modelo.predict_proba(
        X
    )[:, 1]

    # Probabilidades calibradas
    probabilidades_calibradas = calibrador.predict_proba(
        probabilidades_originales.reshape(-1, 1)
    )[:, 1]

    # Evaluacion sin calibrar
    resultados.append(
        evaluar(
            "Logistica_original",
            periodo + "_" + tipo,
            datos,
            probabilidades_originales,
            0.45
        )
    )

    # Evaluacion calibrada
    resultados.append(
        evaluar(
            "Logistica_calibrada",
            periodo + "_" + tipo,
            datos,
            probabilidades_calibradas,
            mejor_umbral
        )
    )


# =====================================================
# 12. GUARDAR RESULTADOS
# =====================================================

tabla = pd.DataFrame(resultados)

ruta = (
    salida /
    "112_sensibilidad_sin_estudiantes_compartidos.csv"
)

tabla.to_csv(ruta, index=False)

print("\nRESULTADOS DEL EXPERIMENTO 112")

print(
    tabla.round(4).to_string(index=False)
)

print("\nARCHIVO GUARDADO EN:")
print(ruta)
