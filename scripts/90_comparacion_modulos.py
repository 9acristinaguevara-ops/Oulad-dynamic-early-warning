
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)

# 1. Rutas
CARPETA = Path("outputs/tables")

ARCHIVO = CARPETA / "89_comparacion_faltantes_predicciones.csv"
SALIDA = CARPETA / "90_comparacion_modulos.csv"

# 2. Cargar las predicciones guardadas
df = pd.read_csv(ARCHIVO)

print("Predicciones cargadas:", len(df))

# 3. Calcular métricas por versión, periodo y módulo
resultados = []

for version in df["version"].unique():
    for periodo in df["periodo"].unique():
        for modulo in sorted(df["code_module"].unique()):

            grupo = df[
                (df["version"] == version) &
                (df["periodo"] == periodo) &
                (df["code_module"] == modulo)
            ].copy()

            if grupo.empty:
                continue

            reales = grupo["real"].astype(int)
            predicciones = grupo["prediccion"].astype(int)
            probabilidades = grupo["probabilidad"]

            resultado = {
                "version": version,
                "periodo": periodo,
                "modulo": modulo,
                "estudiantes": len(grupo),
                "retiradas": int(reales.sum()),
                "alertas": int(predicciones.sum()),
                "precision": precision_score(
                    reales, predicciones, zero_division=0
                ),
                "recall": recall_score(
                    reales, predicciones, zero_division=0
                ),
                "f1": f1_score(
                    reales, predicciones, zero_division=0
                ),
                "accuracy": accuracy_score(
                    reales, predicciones
                ),
                "auc": (
                    roc_auc_score(reales, probabilidades)
                    if reales.nunique() == 2
                    else float("nan")
                )
            }

            resultados.append(resultado)

# 4. Guardar resultados
tabla = pd.DataFrame(resultados)
tabla.to_csv(SALIDA, index=False)

# 5. Mostrar métricas
print("\nRESULTADOS POR MÓDULO")
print(
    tabla[
        [
            "version", "periodo", "modulo",
            "estudiantes", "retiradas", "alertas",
            "precision", "recall", "f1", "auc"
        ]
    ].round(3).to_string(index=False)
)

# 6. Comparar el F1 de las dos versiones
comparacion = tabla.pivot_table(
    index=["periodo", "modulo"],
    columns="version",
    values="f1"
).reset_index()

if "Original" in comparacion.columns and "Semana4_cero" in comparacion.columns:
    comparacion["cambio_f1"] = (
        comparacion["Semana4_cero"] - comparacion["Original"]
    )

    print("\nCAMBIO DE F1 POR MÓDULO")
    print(comparacion.round(3).to_string(index=False))

    comparacion.to_csv(
        CARPETA / "90_cambio_f1_por_modulo.csv",
        index=False
    )

print("\nArchivo guardado:")
print(SALIDA)
