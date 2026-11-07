
import pandas as pd
from pathlib import Path

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "interim" / "early_warning_observations.parquet"
OUTPUT_FILE = BASE_DIR / "data" / "interim" / "prediction_snapshot_week4.parquet"
SUMMARY_FILE = BASE_DIR / "outputs" / "tables" / "prediction_snapshot_summary.csv"
REPORT_FILE = BASE_DIR / "outputs" / "logs" / "prediction_snapshot_report.txt"

SEMANA_CORTE = 4

print("Cargando observaciones temporales...")
df = pd.read_parquet(INPUT_FILE)

# Solo utilizamos información registrada hasta la semana de corte
df = df[df["week"] <= SEMANA_CORTE].copy()

# Ordenar para encontrar la última observación disponible
df = df.sort_values(
    ["code_module", "code_presentation", "id_student", "week"]
)

# Una observación por estudiante y curso
snapshot = df.groupby(
    ["code_module", "code_presentation", "id_student"],
    as_index=False
).tail(1).copy()

# Registrar la semana real de la última observación disponible
snapshot = snapshot.rename(columns={"week": "snapshot_week"})

# Variable objetivo:
# 1 = el estudiante terminó con Withdrawn
# 0 = cualquier otro resultado final
snapshot["target_withdrawn"] = (
    snapshot["target_final_result"] == "Withdrawn"
).astype(int)

# Columnas que identifican al estudiante y al curso.
# Se conservan para seguimiento, pero no serán predictores.
identification_columns = [
    "code_module",
    "code_presentation",
    "id_student",
    "snapshot_week",
    "target_final_result",
    "target_withdrawn"
]

# Variables que podrían utilizarse como predictores.
# El resultado final queda excluido.
candidate_features = [
    "weekly_clicks",
    "active_days",
    "activity_records",
    "clicks_previous_week",
    "clicks_change",
    "clicks_relative_change",
    "clicks_mean_4_weeks",
    "clicks_median_4_weeks",
    "clicks_std_4_weeks",
    "consecutive_week",
    "activity_observed",
    "has_previous_week",
    "weekly_change",
    "activity_decreased"
]

# Conservar solamente las variables que existen en el archivo
available_features = [
    col for col in candidate_features if col in snapshot.columns
]

output_columns = identification_columns + available_features
snapshot = snapshot[output_columns].copy()

# Resumen de las clases
summary = snapshot.groupby(
    ["target_final_result", "target_withdrawn"],
    dropna=False
).agg(
    student_courses=("id_student", "count"),
    mean_snapshot_week=("snapshot_week", "mean")
).reset_index()

# Resumen adicional de la variable objetivo binaria
binary_summary = snapshot["target_withdrawn"].value_counts().sort_index()

# Guardar resultados
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

snapshot.to_parquet(OUTPUT_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("CONJUNTO DE PREDICCION: ALERTA TEMPRANA\n")
    file.write(f"Semana de corte: {SEMANA_CORTE}\n")
    file.write(f"Estudiante-curso incluidos: {len(snapshot)}\n")
    file.write(
        f"Semana real media de observacion: "
        f"{snapshot['snapshot_week'].mean():.2f}\n\n"
    )
    file.write("Distribucion de la variable objetivo:\n")
    file.write(binary_summary.to_string())
    file.write("\n\nVariables candidatas:\n")
    file.write("\n".join(available_features))
    file.write(
        "\n\nLimitacion: solo se incluyen estudiantes con "
        "actividad registrada hasta la semana de corte.\n"
    )
    file.write(
        "El resultado final se conserva como etiqueta, "
        "no como predictor.\n"
    )

print("\nConjunto de prediccion creado.")
print("Estudiantes-curso:", len(snapshot))
print("\nDistribución de resultados:")
print(summary.to_string(index=False))
print("\nDistribución de la variable objetivo (0=no Withdrawn, 1=Withdrawn):")
print(binary_summary.to_string())
print("\nVariables candidatas:", available_features)
print("\nArchivo:", OUTPUT_FILE)
print("Resumen:", SUMMARY_FILE)
print("Informe:", REPORT_FILE)
