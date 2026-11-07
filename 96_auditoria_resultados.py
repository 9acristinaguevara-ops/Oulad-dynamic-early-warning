
import pandas as pd

# Archivos que vamos a revisar
archivos = [
    "outputs/tables/82_model_comparison_detailed.csv",
    "outputs/tables/89_comparacion_faltantes_resultados.csv",
    "outputs/tables/92_comparacion_modelos.csv",
    "outputs/tables/94_comparacion_umbral_modulo.csv"
]

for archivo in archivos:
    print("\n" + "=" * 70)
    print("ARCHIVO:", archivo)
    print("=" * 70)

    try:
        datos = pd.read_csv(archivo)

        print("\nColumnas disponibles:")
        print(datos.columns.tolist())

        print("\nNúmero de filas:", len(datos))

        print("\nContenido:")
        print(datos.to_string(index=False))

    except Exception as error:
        print("No se pudo leer el archivo:", error)

print("\nAUDITORÍA TERMINADA")
