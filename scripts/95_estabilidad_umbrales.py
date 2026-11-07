
import pandas as pd
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score


# 1. Rutas
ENTRADA = Path("data/interim/week4_combined_features_v4.parquet")
SALIDA = Path("outputs/tables")
SALIDA.mkdir(parents=True, exist_ok=True)


# 2. Cargar dataset y resultados del script 94
df = pd.read_parquet(ENTRADA)

resultados = pd.read_csv(
    SALIDA / "94_comparacion_umbral_modulo.csv"
)

# En el script 94, los umbrales por módulo se eligieron con 2013J.
umbrales = {
    "AAA": 0.40,
    "BBB": 0.30,
    "CCC": 0.20,
    "DDD": 0.20,
    "EEE": 0.50,
    "FFF": 0.20,
    "GGG": 0.50
}


# 3. Resumen de umbrales
print("\nUMBRALES UTILIZADOS")
for modulo, umbral in umbrales.items():
    print(modulo, ":", umbral)


# 4. Comparar métricas por módulo entre los dos períodos
print("\nESTABILIDAD DEL F1 POR MÓDULO")

tabla = resultados.pivot_table(
    index="modulo",
    columns=["periodo", "metodo"],
    values="f1"
)

print(tabla.round(3).to_string())


# 5. Calcular diferencia de F1 entre umbral por módulo y global
print("\nDIFERENCIA DE F1: POR MÓDULO MENOS GLOBAL")

for periodo in ["2014B", "2014J"]:

    datos = resultados[
        resultados["periodo"] == periodo
    ]

    globales = datos[
        datos["metodo"] == "Global_020"
    ].set_index("modulo")

    especificos = datos[
        datos["metodo"] == "Por_modulo"
    ].set_index("modulo")

    comparacion = []

    for modulo in sorted(globales.index):

        f1_global = globales.loc[modulo, "f1"]
        f1_modulo = especificos.loc[modulo, "f1"]

        comparacion.append({
            "modulo": modulo,
            "f1_global": f1_global,
            "f1_por_modulo": f1_modulo,
            "diferencia_f1": f1_modulo - f1_global
        })

    tabla_diferencias = pd.DataFrame(comparacion)

    print("\nPeríodo:", periodo)
    print(tabla_diferencias.round(3).to_string(index=False))


# 6. Guardar los resultados
tabla.to_csv(SALIDA / "95_estabilidad_f1.csv")

print("\nArchivo guardado:")
print(SALIDA / "95_estabilidad_f1.csv")
