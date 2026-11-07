
from pathlib import Path

import pandas as pd


# ==========================================
# 1. RUTAS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "temporal_student_activity.parquet"
)

OUTPUT_DIR = BASE_DIR / "data" / "interim"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
LOGS_DIR = BASE_DIR / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CARGAR DATOS
# ==========================================

print("Cargando actividad temporal...")
data = pd.read_parquet(INPUT_FILE)

keys = [
    "code_module",
    "code_presentation",
    "id_student"
]

data = data.sort_values(
    keys + ["week"]
).reset_index(drop=True)


# ==========================================
# 3. CREAR VARIABLES HISTÓRICAS
# ==========================================

print("Calculando variables históricas...")

# Media móvil retrospectiva de cuatro registros observados.
# No utiliza semanas futuras.
data["mean_clicks_past_4"] = (
    data.groupby(keys)["weekly_clicks"]
    .transform(
        lambda values: values.rolling(
            window=4,
            min_periods=1
        ).mean()
    )
)

# Mediana retrospectiva de cuatro registros observados
data["median_clicks_past_4"] = (
    data.groupby(keys)["weekly_clicks"]
    .transform(
        lambda values: values.rolling(
            window=4,
            min_periods=1
        ).median()
    )
)

# Variabilidad retrospectiva
data["std_clicks_past_4"] = (
    data.groupby(keys)["weekly_clicks"]
    .transform(
        lambda values: values.rolling(
            window=4,
            min_periods=2
        ).std()
    )
)

# Clics de la semana inmediatamente anterior
data["previous_clicks"] = (
    data.groupby(keys)["weekly_clicks"].shift(1)
)

previous_week = data.groupby(keys)["week"].shift(1)

data["consecutive_week"] = (
    data["week"] == previous_week + 1
)

# Diferencia semanal válida solo para semanas consecutivas
data["weekly_change"] = (
    data["weekly_clicks"] - data["previous_clicks"]
)

data.loc[
    ~data["consecutive_week"],
    "weekly_change"
] = pd.NA

# Indicador de descenso respecto a la semana anterior
data["activity_decreased"] = pd.NA

valid = (
    data["consecutive_week"]
    & data["previous_clicks"].notna()
)

data.loc[valid, "activity_decreased"] = (
    data.loc[valid, "weekly_clicks"]
    < data.loc[valid, "previous_clicks"]
).astype(int)

# ==========================================
# 4. CREAR TABLA DE ALERTA TEMPRANA
# ==========================================

# Cada fila representa a un estudiante-curso en una semana.
# El resultado final se conserva únicamente como etiqueta.
predictors = [
    "code_module",
    "code_presentation",
    "id_student",
    "week",
    "weekly_clicks",
    "active_days",
    "activity_records",
    "mean_clicks_past_4",
    "median_clicks_past_4",
    "std_clicks_past_4",
    "previous_clicks",
    "weekly_change",
    "activity_decreased",
    "final_result"
]

early_data = data[predictors].copy()

# Eliminar semanas sin historial consecutivo para evaluar
# el cambio semanal. Se mantienen otras variables históricas.
early_data["has_previous_week"] = (
    data["consecutive_week"]
    & data["previous_clicks"].notna()
).to_numpy()

# El resultado final no debe incluirse entre las variables X.
# Se mantiene separado como etiqueta de evaluación.
early_data = early_data.rename(
    columns={"final_result": "target_final_result"}
)

output_file = (
    OUTPUT_DIR / "early_warning_observations.parquet"
)

early_data.to_parquet(
    output_file,
    index=False
)


# ==========================================
# 5. GUARDAR UN RESUMEN DE COBERTURA
# ==========================================

coverage = (
    early_data.groupby("week", as_index=False)
    .agg(
        observations=("id_student", "size"),
        students=("id_student", "nunique"),
        mean_clicks=("weekly_clicks", "mean"),
        median_clicks=("weekly_clicks", "median"),
        observations_with_previous_week=(
            "has_previous_week",
            "sum"
        )
    )
)

coverage.to_csv(
    TABLES_DIR / "early_warning_weekly_coverage.csv",
    index=False
)


# ==========================================
# 6. RESUMEN DE ETIQUETAS
# ==========================================

labels = (
    early_data.groupby(
        "target_final_result",
        dropna=False,
        as_index=False
    )
    .agg(
        observations=("id_student", "size"),
        students=("id_student", "nunique")
    )
)

labels.to_csv(
    TABLES_DIR / "early_warning_label_summary.csv",
    index=False
)


# ==========================================
# 7. INFORME
# ==========================================

with open(
    LOGS_DIR / "early_warning_dataset_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("CONJUNTO TEMPORAL DE ALERTA TEMPRANA\n")
    file.write("====================================\n\n")

    file.write(f"Observaciones: {len(early_data)}\n")
    file.write(
        f"Estudiantes únicos: {early_data['id_student'].nunique()}\n"
    )
    file.write(
        f"Semana mínima: {early_data['week'].min()}\n"
    )
    file.write(
        f"Semana máxima: {early_data['week'].max()}\n\n"
    )

    file.write("DISTRIBUCIÓN DE RESULTADOS FINALES\n")
    file.write(labels.to_string(index=False))
    file.write("\n\n")

    file.write(
        "El resultado final se conserva como etiqueta, "
        "no como predictor.\n"
    )

    file.write(
        "Este archivo prepara observaciones históricas, "
        "pero no constituye todavía una evaluación predictiva.\n"
    )


print("\nConjunto temporal de alerta temprana creado.")
print("Archivo:", output_file)
print(
    "Cobertura semanal:",
    TABLES_DIR / "early_warning_weekly_coverage.csv"
)
print(
    "Resumen de etiquetas:",
    TABLES_DIR / "early_warning_label_summary.csv"
)
print(
    "Informe:",
    LOGS_DIR / "early_warning_dataset_report.txt"
)
