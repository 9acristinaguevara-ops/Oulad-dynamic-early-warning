
import pandas as pd
from pathlib import Path

BASE = Path(".")
INTERIM = BASE / "data" / "interim"

# Leer datos semanales y conjunto corregido
actividad = pd.read_parquet(
    INTERIM / "weekly_activity.parquet"
)

datos = pd.read_parquet(
    INTERIM / "corrected_week4_final_dataset.parquet"
)

claves = [
    "code_module",
    "code_presentation",
    "id_student"
]

# Usar solo semanas 0 a 4
actividad = actividad[
    actividad["week"].between(0, 4)
].copy()

# Crear una tabla por estudiante y semana
clics = actividad.pivot_table(
    index=claves,
    columns="week",
    values="weekly_clicks",
    aggfunc="sum"
)

dias = actividad.pivot_table(
    index=claves,
    columns="week",
    values="active_days",
    aggfunc="sum"
)

registros = actividad.pivot_table(
    index=claves,
    columns="week",
    values="activity_records",
    aggfunc="sum"
)

# Renombrar columnas de clics
clics.columns = [
    "clics_semana_" + str(int(c))
    for c in clics.columns
]

dias.columns = [
    "dias_semana_" + str(int(c))
    for c in dias.columns
]

registros.columns = [
    "registros_semana_" + str(int(c))
    for c in registros.columns
]

# Unir las tablas semanales
features = clics.join(dias).join(registros).reset_index()

# Añadir nuevas variables
columnas_clics = [
    "clics_semana_" + str(i)
    for i in range(5)
]

columnas_dias = [
    "dias_semana_" + str(i)
    for i in range(5)
]

columnas_registros = [
    "registros_semana_" + str(i)
    for i in range(5)
]

# Contar semanas con registros de actividad
features["semanas_con_actividad"] = (
    features[columnas_clics].notna().sum(axis=1)
)

# Sumar clics solo si existen datos en alguna semana
features["clics_acumulados_0_4"] = (
    features[columnas_clics].sum(axis=1, min_count=1)
)

# Promedio semanal, sin convertir semanas ausentes en ceros
features["promedio_clics_0_4"] = (
    features[columnas_clics].mean(axis=1)
)

# Días activos acumulados
features["dias_activos_0_4"] = (
    features[columnas_dias].sum(axis=1, min_count=1)
)

# Clics recientes: semanas 3 y 4
features["clics_recientes_3_4"] = (
    features[["clics_semana_3", "clics_semana_4"]]
    .sum(axis=1, min_count=1)
)

# Clics anteriores: semanas 0, 1 y 2
features["clics_anteriores_0_2"] = (
    features[["clics_semana_0", "clics_semana_1",
              "clics_semana_2"]]
    .sum(axis=1, min_count=1)
)

# Unir con las etiquetas corregidas
datos = datos.merge(
    features,
    on=claves,
    how="left",
    validate="one_to_one"
)

# Guardar un nuevo archivo
salida = INTERIM / "week4_enriched_features.parquet"
datos.to_parquet(salida, index=False)

print("CONJUNTO ENRIQUECIDO CREADO")
print("Filas:", len(datos))
print("Nuevas variables:")
print([
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "promedio_clics_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2"
])
print("\nValores faltantes de nuevas variables:")
print(datos[
    [
        "semanas_con_actividad",
        "clics_acumulados_0_4",
        "promedio_clics_0_4",
        "dias_activos_0_4",
        "clics_recientes_3_4",
        "clics_anteriores_0_2"
    ]
].isna().sum().to_string())
print("\nArchivo guardado:", salida)
