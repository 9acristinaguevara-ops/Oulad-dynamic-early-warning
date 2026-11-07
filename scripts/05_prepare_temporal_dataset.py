
from pathlib import Path

import pandas as pd


# ==========================================
# 1. RUTAS DEL PROYECTO
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

ACTIVITY_FILE = (
    BASE_DIR / "data" / "interim" / "weekly_activity.parquet"
)

STUDENT_FILE = (
    BASE_DIR / "data" / "raw" / "csv" / "studentInfo.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "interim"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
LOGS_DIR = BASE_DIR / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CARGAR LOS DATOS
# ==========================================

print("Cargando actividad semanal...")
activity = pd.read_parquet(ACTIVITY_FILE)

print("Cargando información de estudiantes...")
students = pd.read_csv(STUDENT_FILE)

keys = [
    "code_module",
    "code_presentation",
    "id_student"
]


# ==========================================
# 3. UNIR ACTIVIDAD Y RESULTADOS
# ==========================================

data = activity.merge(
    students,
    on=keys,
    how="left",
    validate="many_to_one"
)

data = data.sort_values(
    keys + ["week"]
).reset_index(drop=True)

print("Registros cargados:", len(data))


# ==========================================
# 4. CREAR INDICADORES TEMPORALES
# ==========================================

# Clics de la semana anterior
data["clicks_previous_week"] = (
    data.groupby(keys)["weekly_clicks"].shift(1)
)

# Diferencia entre clics y semana anterior
data["clicks_change"] = (
    data["weekly_clicks"] - data["clicks_previous_week"]
)

# Cambio relativo de actividad.
# Se calcula solamente cuando la semana anterior tiene
# un valor positivo y consecutivo en el calendario.
previous_week = data.groupby(keys)["week"].shift(1)

consecutive_week = (
    data["week"] == previous_week + 1
)

valid_previous = (
    data["clicks_previous_week"].notna()
    & (data["clicks_previous_week"] > 0)
    & consecutive_week
)

data["clicks_relative_change"] = pd.NA

data.loc[valid_previous, "clicks_relative_change"] = (
    (
        data.loc[valid_previous, "weekly_clicks"]
        - data.loc[valid_previous, "clicks_previous_week"]
    )
    / data.loc[valid_previous, "clicks_previous_week"]
)


# ==========================================
# 5. PROMEDIO MÓVIL DE CUATRO SEMANAS
# ==========================================

data["clicks_mean_4_weeks"] = (
    data.groupby(keys)["weekly_clicks"]
    .transform(
        lambda values: values.rolling(
            window=4,
            min_periods=1
        ).mean()
    )
)


# ==========================================
# 6. IDENTIFICAR SEMANAS CON HUECOS
# ==========================================

previous_week = data.groupby(keys)["week"].shift(1)

data["consecutive_week"] = (
    data["week"] == previous_week + 1
)

# Una fila por semana significa que existe actividad
# registrada en esa semana.
# Las semanas ausentes no se convierten en cero clics.

data["activity_observed"] = 1


# ==========================================
# 7. GUARDAR EL CONJUNTO TEMPORAL
# ==========================================

output_columns = keys + [
    "week",
    "weekly_clicks",
    "active_days",
    "activity_records",
    "clicks_previous_week",
    "clicks_change",
    "clicks_relative_change",
    "clicks_mean_4_weeks",
    "consecutive_week",
    "activity_observed",
    "final_result"
]

temporal_data = data[output_columns].copy()

parquet_output = (
    OUTPUT_DIR / "temporal_student_activity.parquet"
)

csv_output = (
    OUTPUT_DIR / "temporal_student_activity.csv"
)

temporal_data.to_parquet(
    parquet_output,
    index=False
)

temporal_data.to_csv(
    csv_output,
    index=False
)


# ==========================================
# 8. RESUMEN DE CALIDAD
# ==========================================

summary = {
    "total_records": len(temporal_data),
    "unique_students": temporal_data["id_student"].nunique(),
    "minimum_week": temporal_data["week"].min(),
    "maximum_week": temporal_data["week"].max(),
    "missing_previous_clicks": int(
        temporal_data["clicks_previous_week"].isna().sum()
    ),
    "non_consecutive_records": int(
        (~temporal_data["consecutive_week"])
        .sum()
    ),
    "missing_final_result": int(
        temporal_data["final_result"].isna().sum()
    )
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["indicator", "value"]
)

summary_df.to_csv(
    TABLES_DIR / "temporal_dataset_quality.csv",
    index=False
)

with open(
    LOGS_DIR / "temporal_dataset_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("REPORTE DEL CONJUNTO TEMPORAL OULAD\n")
    file.write("===================================\n\n")

    for indicator, value in summary.items():
        file.write(f"{indicator}: {value}\n")


# ==========================================
# 9. RESULTADOS
# ==========================================

print("\nPreparación temporal terminada.")
print("Registros:", len(temporal_data))
print("Estudiantes:", temporal_data["id_student"].nunique())
print("Archivo Parquet:", parquet_output)
print("Archivo CSV:", csv_output)
print(
    "Resumen de calidad:",
    TABLES_DIR / "temporal_dataset_quality.csv"
)
print(
    "Informe:",
    LOGS_DIR / "temporal_dataset_report.txt"
)

print(
    "\nNota: las semanas ausentes no se han convertido "
    "en semanas con cero clics."
)
print(
    "El resultado final se conserva para evaluación, "
    "no debe utilizarse como predictor."
)
