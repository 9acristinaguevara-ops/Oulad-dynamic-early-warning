
import pandas as pd

archivos = [
    "outputs/tables/83_importancia_permutacion_model_b.csv",
    "outputs/tables/84_importancia_temporal_model_b.csv",
    "outputs/tables/85_comparacion_variables.csv",
    "outputs/tables/85_metricas_temporales.csv",
    "outputs/tables/86_promedios_por_objetivo.csv",
    "outputs/tables/88_auditoria_actividad_por_periodo.csv",
    "outputs/tables/91_faltantes_por_periodo.csv",
    "outputs/tables/91_resumen_modulo_periodo.csv"
]

for archivo in archivos:
    print("\n" + "=" * 75)
    print("ARCHIVO:", archivo)
    print("=" * 75)

    try:
        datos = pd.read_csv(archivo)

        print("Columnas:", datos.columns.tolist())
        print("Filas:", len(datos))

        print(datos.to_string(index=False))

    except Exception as error:
        print("Error al leer el archivo:", error)

print("\nAUDITORÍA DE VARIABLES Y TIEMPO TERMINADA")
