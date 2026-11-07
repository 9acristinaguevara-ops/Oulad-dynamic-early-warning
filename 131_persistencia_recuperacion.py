
# 131_persistencia_recuperacion.py

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import mannwhitneyu

# =====================================================
# 1. RUTAS
# =====================================================

BASE = Path(__file__).resolve().parent.parent

ARCHIVO = (
    BASE / "data" / "interim" /
    "weekly_panel_weeks0_4.parquet"
)

SALIDA_TABLAS = BASE / "outputs" / "tables"
SALIDA_FIGURAS = BASE / "outputs" / "figures"

SALIDA_TABLAS.mkdir(parents=True, exist_ok=True)
SALIDA_FIGURAS.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. CARGAR DATOS
# =====================================================

df = pd.read_parquet(ARCHIVO)

columnas = [
    "clics_semana_0",
    "clics_semana_1",
    "clics_semana_2",
    "clics_semana_3",
    "clics_semana_4"
]

# Trabajamos con estudiantes que tienen
# actividad registrada en las cinco semanas.

df = df[
    df[columnas].notna().all(axis=1)
].copy()

print("DATOS CARGADOS")
print("============================")
print("Trayectorias completas:", len(df))

# =====================================================
# 3. CALCULAR CAMBIOS SEMANALES
# =====================================================

resultados = []

for _, fila in df.iterrows():

    clics = [
        fila["clics_semana_0"],
        fila["clics_semana_1"],
        fila["clics_semana_2"],
        fila["clics_semana_3"],
        fila["clics_semana_4"]
    ]

    # Diferencia entre semanas consecutivas
    cambios = []

    for i in range(1, len(clics)):
        diferencia = clics[i] - clics[i - 1]
        cambios.append(diferencia)

    # Clasificar cada cambio
    direcciones = []

    for cambio in cambios:

        if cambio < 0:
            direcciones.append(-1)

        elif cambio > 0:
            direcciones.append(1)

        else:
            direcciones.append(0)

    disminuciones = direcciones.count(-1)
    aumentos = direcciones.count(1)
    estabilidad = direcciones.count(0)

    # Disminuciones consecutivas:
    # una disminución seguida de otra disminución
    persistencia_descenso = 0

    # Recuperaciones:
    # una disminución seguida de un aumento
    recuperaciones = 0

    for i in range(len(direcciones) - 1):

        if direcciones[i] == -1:

            if direcciones[i + 1] == -1:
                persistencia_descenso += 1

            elif direcciones[i + 1] == 1:
                recuperaciones += 1

    # Proporción de pares consecutivos
    # que presentan dos disminuciones
    tasa_persistencia = (
        persistencia_descenso / 3
    )

    resultados.append({

        "code_module": fila["code_module"],
        "code_presentation": fila["code_presentation"],
        "id_student": fila["id_student"],

        "target_future_withdrawal":
            fila["target_future_withdrawal"],

        "disminuciones": disminuciones,
        "aumentos": aumentos,
        "semanas_estables": estabilidad,

        "persistencia_descenso":
            persistencia_descenso,

        "recuperaciones": recuperaciones,

        "tasa_persistencia":
            tasa_persistencia

    })

# Crear un DataFrame con los resultados
trayectorias = pd.DataFrame(resultados)

# =====================================================
# 4. RESUMEN POR GRUPO
# =====================================================

resumen = (
    trayectorias
    .groupby("target_future_withdrawal")
    .agg(
        estudiantes=("id_student", "count"),

        media_disminuciones=("disminuciones", "mean"),
        mediana_disminuciones=("disminuciones", "median"),

        media_aumentos=("aumentos", "mean"),
        mediana_aumentos=("aumentos", "median"),

        media_persistencia=("persistencia_descenso", "mean"),
        mediana_persistencia=("persistencia_descenso", "median"),

        media_recuperaciones=("recuperaciones", "mean"),
        mediana_recuperaciones=("recuperaciones", "median"),

        media_tasa_persistencia=("tasa_persistencia", "mean")
    )
    .reset_index()
)

resumen["grupo_resultado"] = (
    resumen["target_future_withdrawal"]
    .map({
        0: "No abandono",
        1: "Abandono futuro"
    })
)

print("\nRESUMEN DE DINAMICA INDIVIDUAL")
print("============================")
print(resumen.round(3).to_string(index=False))

# =====================================================
# 5. FRECUENCIAS DE PERSISTENCIA Y RECUPERACION
# =====================================================

print("\nFRECUENCIAS DE PERSISTENCIA")
print("============================")

frecuencia_persistencia = pd.crosstab(
    trayectorias["target_future_withdrawal"],
    trayectorias["persistencia_descenso"],
    normalize="index"
) * 100

print(frecuencia_persistencia.round(2))

print("\nFRECUENCIAS DE RECUPERACION")
print("============================")

frecuencia_recuperacion = pd.crosstab(
    trayectorias["target_future_withdrawal"],
    trayectorias["recuperaciones"],
    normalize="index"
) * 100

print(frecuencia_recuperacion.round(2))

# =====================================================
# 6. COMPARACIONES ESTADISTICAS
# =====================================================

print("\nPRUEBAS MANN-WHITNEY")
print("============================")

variables = [
    "disminuciones",
    "aumentos",
    "persistencia_descenso",
    "recuperaciones",
    "tasa_persistencia"
]

for variable in variables:

    grupo0 = trayectorias[
        trayectorias["target_future_withdrawal"] == 0
    ][variable]

    grupo1 = trayectorias[
        trayectorias["target_future_withdrawal"] == 1
    ][variable]

    prueba = mannwhitneyu(
        grupo0,
        grupo1,
        alternative="two-sided"
    )

    print(
        variable,
        "| U =", round(prueba.statistic, 2),
        "| p =", prueba.pvalue
    )

# =====================================================
# 7. GUARDAR RESULTADOS
# =====================================================

trayectorias.to_csv(
    SALIDA_TABLAS / "131_metricas_individuales.csv",
    index=False
)

resumen.to_csv(
    SALIDA_TABLAS / "131_resumen_dinamica.csv",
    index=False
)

frecuencia_persistencia.to_csv(
    SALIDA_TABLAS / "131_frecuencia_persistencia.csv"
)

frecuencia_recuperacion.to_csv(
    SALIDA_TABLAS / "131_frecuencia_recuperacion.csv"
)

# =====================================================
# 8. GRAFICO DE DISMINUCIONES
# =====================================================

plt.figure(figsize=(8, 5))

datos = [
    trayectorias[
        trayectorias["target_future_withdrawal"] == 0
    ]["disminuciones"],

    trayectorias[
        trayectorias["target_future_withdrawal"] == 1
    ]["disminuciones"]
]

plt.boxplot(
    datos,
    tick_labels=["No abandono", "Abandono futuro"]
)

plt.xlabel("Resultado")
plt.ylabel("Número de disminuciones")
plt.title("Disminuciones de actividad por estudiante")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    SALIDA_FIGURAS / "131_disminuciones.png",
    dpi=300
)

plt.close()

# =====================================================
# 9. GRAFICO DE PERSISTENCIA
# =====================================================

plt.figure(figsize=(8, 5))

datos = [
    trayectorias[
        trayectorias["target_future_withdrawal"] == 0
    ]["persistencia_descenso"],

    trayectorias[
        trayectorias["target_future_withdrawal"] == 1
    ]["persistencia_descenso"]
]

plt.boxplot(
    datos,
    tick_labels=["No abandono", "Abandono futuro"]
)

plt.xlabel("Resultado")
plt.ylabel("Pares de disminuciones consecutivas")
plt.title("Persistencia de los descensos de actividad")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    SALIDA_FIGURAS / "131_persistencia.png",
    dpi=300
)

plt.close()

# =====================================================
# 10. GRAFICO DE RECUPERACIONES
# =====================================================

plt.figure(figsize=(8, 5))

datos = [
    trayectorias[
        trayectorias["target_future_withdrawal"] == 0
    ]["recuperaciones"],

    trayectorias[
        trayectorias["target_future_withdrawal"] == 1
    ]["recuperaciones"]
]

plt.boxplot(
    datos,
    tick_labels=["No abandono", "Abandono futuro"]
)

plt.xlabel("Resultado")
plt.ylabel("Número de recuperaciones")
plt.title("Recuperaciones de actividad por estudiante")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    SALIDA_FIGURAS / "131_recuperaciones.png",
    dpi=300
)

plt.close()

# =====================================================
# 11. FINALIZAR
# =====================================================

print("\nARCHIVOS GUARDADOS")
print("============================")

print("131_metricas_individuales.csv")
print("131_resumen_dinamica.csv")
print("131_frecuencia_persistencia.csv")
print("131_frecuencia_recuperacion.csv")

print("\nFiguras:")
print("131_disminuciones.png")
print("131_persistencia.png")
print("131_recuperaciones.png")

print("\nANALISIS FINALIZADO")
