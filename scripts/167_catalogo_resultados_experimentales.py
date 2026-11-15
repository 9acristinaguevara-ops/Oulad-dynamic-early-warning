from pathlib import Path
import hashlib
import pandas as pd

# Experimento 167: inventario de resultados experimentales.
# No modifica ni sobrescribe los CSV originales.

BASE = Path(__file__).resolve().parents[1]
CARPETA_TABLAS = BASE / "outputs" / "tables"
CARPETA_SALIDA = BASE / "outputs" / "tables"

PREFIJOS = tuple(str(n) + "_" for n in range(107, 167))

def calcular_sha256(ruta):
    sha = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        while True:
            bloque = archivo.read(1024 * 1024)
            if not bloque:
                break
            sha.update(bloque)
    return sha.hexdigest()

def leer_csv(ruta):
    try:
        datos = pd.read_csv(ruta)
        return datos, "OK"
    except UnicodeDecodeError:
        try:
            datos = pd.read_csv(ruta, encoding="latin-1")
            return datos, "OK_LATIN1"
        except Exception as error:
            return None, "ERROR: " + str(error)
    except Exception as error:
        return None, "ERROR: " + str(error)

def main():
    archivos = sorted(
        ruta for ruta in CARPETA_TABLAS.glob("*.csv")
        if ruta.name.startswith(PREFIJOS)
    )

    filas = []

    for ruta in archivos:
        datos, estado = leer_csv(ruta)

        if datos is None:
            filas.append({
                "archivo": ruta.name,
                "experimento": ruta.name.split("_")[0],
                "filas": None,
                "columnas": None,
                "nombres_columnas": "",
                "estado_lectura": estado,
                "bytes": ruta.stat().st_size,
                "sha256": calcular_sha256(ruta)
            })
        else:
            filas.append({
                "archivo": ruta.name,
                "experimento": ruta.name.split("_")[0],
                "filas": len(datos),
                "columnas": len(datos.columns),
                "nombres_columnas": " | ".join(str(c) for c in datos.columns),
                "estado_lectura": estado,
                "bytes": ruta.stat().st_size,
                "sha256": calcular_sha256(ruta)
            })

    catalogo = pd.DataFrame(filas)
    salida = CARPETA_SALIDA / "167_catalogo_resultados_experimentales.csv"
    catalogo.to_csv(salida, index=False, encoding="utf-8-sig")

    resumen = (
        catalogo.groupby("experimento", as_index=False)
        .agg(
            archivos=("archivo", "count"),
            archivos_legibles=("estado_lectura", lambda x: x.str.startswith("OK").sum()),
            archivos_con_error=("estado_lectura", lambda x: x.str.startswith("ERROR").sum())
        )
    )
    salida_resumen = CARPETA_SALIDA / "167_resumen_catalogo_experimental.csv"
    resumen.to_csv(salida_resumen, index=False, encoding="utf-8-sig")

    print("\nEXPERIMENTO 167: CATALOGO DE RESULTADOS")
    print("Carpeta revisada:", CARPETA_TABLAS)
    print("Archivos CSV encontrados:", len(catalogo))
    print("Archivos legibles:", catalogo["estado_lectura"].str.startswith("OK").sum())
    print("Archivos con error:", catalogo["estado_lectura"].str.startswith("ERROR").sum())
    print("\nRESUMEN POR EXPERIMENTO")
    print(resumen.to_string(index=False))
    print("\nARCHIVOS GENERADOS")
    print(salida)
    print(salida_resumen)

if __name__ == "__main__":
    main()
