
import pandas as pd

# 1. Cargar resultados del script 67
ruta = (
    "outputs/tables/"
    "threshold_comparison_by_module_period_v1.csv"
)

df = pd.read_csv(ruta)

# 2. Mostrar el mejor umbral por modelo, periodo y módulo
print("MEJORES UMBRALES POR MODELO, PERIODO Y MODULO")
print()

resultados = []

for modelo in df["modelo"].unique():
    for periodo in sorted(df["periodo"].unique()):
        for modulo in sorted(df["modulo"].unique()):

            grupo = df[
                (df["modelo"] == modelo) &
                (df["periodo"] == periodo) &
                (df["modulo"] == modulo)
            ]

            if len(grupo) == 0:
                continue

            mejor = grupo.loc[grupo["f1"].idxmax()]

            print(
                "Modelo:", modelo,
                "| Periodo:", periodo,
                "| Modulo:", modulo,
                "| Mejor umbral:", mejor["umbral"],
                "| F1:", round(mejor["f1"], 3),
                "| Precision:", round(mejor["precision"], 3),
                "| Recall:", round(mejor["recall"], 3),
                "| Alertas:", int(mejor["alertas"])
            )

            resultados.append({
                "modelo": modelo,
                "periodo": periodo,
                "modulo": modulo,
                "mejor_umbral": mejor["umbral"],
                "f1": mejor["f1"],
                "precision": mejor["precision"],
                "recall": mejor["recall"],
                "alertas": int(mejor["alertas"])
            })


# 3. Guardar resumen
resumen = pd.DataFrame(resultados)

salida = (
    "outputs/tables/"
    "best_thresholds_by_module_period_v1.csv"
)

resumen.to_csv(salida, index=False)

print()
print("Archivo guardado en:", salida)
print("SCRIPT FINALIZADO")
