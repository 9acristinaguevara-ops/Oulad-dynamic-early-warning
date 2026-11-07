
# 130_analisis_trayectorias_temporales.py

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

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

columnas_clicks = [
    "clics_semana_0",
    "clics_semana_1",
    "clics_semana_2",
    "clics_semana_3",
    "clics_semana_4"
]

print("DATOS CARGADOS")
print("============================")
print("Registros:", len(df))

# =====================================================
# 3. TRANSFORMAR A FORMATO LONGITUDINAL
# =====================================================

df_largo = df.melt(
    id_vars=[
        "code_module",
        "code_presentation",
        "id_student",
        "target_future_withdrawal"
    ],
    value_vars=columnas_clicks,
    var_name="semana",
    value_name="clics"
)

# Extraer el número de semana
df_largo["semana"] = (
    df_largo["semana"]
    .str.extract(r"(\d+)")
    .astype(int)
)

# Eliminar únicamente las semanas sin registro
# No se convierten en cero
df_largo = df_largo.dropna(
    subset=["clics"]
).copy()

df_largo["grupo_resultado"] = (
    df_largo["target_future_withdrawal"]
    .map({
        0: "No abandono",
        1: "Abandono futuro"
    })
)

# =====================================================
# 4. RESUMEN DE ACTIVIDAD POR SEMANA
# =====================================================

resumen = (
    df_largo
    .groupby(
        ["target_future_withdrawal", "semana"]
    )
    .agg(
        estudiantes=("id_student", "nunique"),
        media_clicks=("clics", "mean"),
        mediana_clicks=("clics", "median"),
        desviacion=("clics", "std")
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

print("\nACTIVIDAD SEMANAL POR RESULTADO")
print("============================")
print(
    resumen.round(2).to_string(index=False)
)

# =====================================================
# 5. PORCENTAJE DE ESTUDIANTES CON ACTIVIDAD
# =====================================================

total_grupo = (
    df.groupby("target_future_withdrawal")
    .size()
    .to_dict()
)

resumen["total_grupo"] = (
    resumen["target_future_withdrawal"]
    .map(total_grupo)
)

resumen["porcentaje_con_actividad"] = (
    resumen["estudiantes"] /
    resumen["total_grupo"] * 100
)

print("\nCOBERTURA DE ACTIVIDAD SEMANAL")
print("============================")

print(
    resumen[
        [
            "grupo_resultado",
            "semana",
            "estudiantes",
            "total_grupo",
            "porcentaje_con_actividad"
        ]
    ].round(2).to_string(index=False)
)

# =====================================================
# 6. IDENTIFICAR TRAYECTORIAS COMPLETAS
# =====================================================

df["semanas_observadas"] = (
    df[columnas_clicks].notna().sum(axis=1)
)

df_completo = df[
    df["semanas_observadas"] == 5
].copy()

print("\nTRAYECTORIAS COMPLETAS")
print("============================")
print("Estudiantes con cinco semanas:", len(df_completo))

print("\nDistribución por resultado:")

print(
    df_completo["target_future_withdrawal"]
    .value_counts()
    .sort_index()
)

# =====================================================
# 7. RESUMEN DE TRAYECTORIAS COMPLETAS
# =====================================================

resumen_completo = (
    df_completo
    .groupby("target_future_withdrawal")[columnas_clicks]
    .agg(["mean", "median", "std"])
)

print("\nESTADISTICAS DE TRAYECTORIAS COMPLETAS")
print("============================")

print(resumen_completo.round(2).to_string())

# =====================================================
# 8. GUARDAR TABLAS
# =====================================================

resumen.to_csv(
    SALIDA_TABLAS / "130_actividad_semanal_por_resultado.csv",
    index=False
)

resumen_completo.to_csv(
    SALIDA_TABLAS / "130_trayectorias_completas.csv"
)

# =====================================================
# 9. GRAFICO DE ACTIVIDAD MEDIA
# =====================================================

plt.figure(figsize=(9, 5))

for resultado, nombre in [
    (0, "No abandono"),
    (1, "Abandono futuro")
]:

    datos = resumen[
        resumen["target_future_withdrawal"] == resultado
    ]

    plt.plot(
        datos["semana"],
        datos["media_clicks"],
        marker="o",
        linewidth=2,
        label=nombre
    )

plt.xlabel("Semana")
plt.ylabel("Promedio de clics")
plt.title("Trayectorias medias de actividad académica")
plt.xticks([0, 1, 2, 3, 4])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    SALIDA_FIGURAS / "130_trayectorias_medias.png",
    dpi=300
)

plt.close()

# =====================================================
# 10. GRAFICO DE MEDIANAS
# =====================================================

plt.figure(figsize=(9, 5))

for resultado, nombre in [
    (0, "No abandono"),
    (1, "Abandono futuro")
]:

    datos = resumen[
        resumen["target_future_withdrawal"] == resultado
    ]

    plt.plot(
        datos["semana"],
        datos["mediana_clicks"],
        marker="o",
        linewidth=2,
        label=nombre
    )

plt.xlabel("Semana")
plt.ylabel("Mediana de clics")
plt.title("Trayectorias medianas de actividad académica")
plt.xticks([0, 1, 2, 3, 4])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    SALIDA_FIGURAS / "130_trayectorias_medianas.png",
    dpi=300
)

plt.close()

# =====================================================
# 11. GRAFICO DE COBERTURA
# =====================================================

plt.figure(figsize=(9, 5))

for resultado, nombre in [
    (0, "No abandono"),
    (1, "Abandono futuro")
]:

    datos = resumen[
        resumen["target_future_withdrawal"] == resultado
    ]

    plt.plot(
        datos["semana"],
        datos["porcentaje_con_actividad"],
        marker="o",
        linewidth=2,
        label=nombre
    )

plt.xlabel("Semana")
plt.ylabel("Estudiantes con actividad (%)")
plt.title("Cobertura de registros semanales")
plt.xticks([0, 1, 2, 3, 4])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    SALIDA_FIGURAS / "130_cobertura_semanal.png",
    dpi=300
)

plt.close()

# =====================================================
# 12. FINALIZAR
# =====================================================

print("\nARCHIVOS GUARDADOS")
print("============================")

print("Tablas:")
print("130_actividad_semanal_por_resultado.csv")
print("130_trayectorias_completas.csv")

print("\nFiguras:")
print("130_trayectorias_medias.png")
print("130_trayectorias_medianas.png")
print("130_cobertura_semanal.png")

print("\nANALISIS FINALIZADO")
