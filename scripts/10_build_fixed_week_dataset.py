
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "interim" /
    "prediction_snapshot_week4.parquet"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "interim" /
    "fixed_week4_dataset.parquet"
)

SUMMARY_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "fixed_week4_summary.csv"
)

REPORT_FILE = (
    BASE_DIR / "outputs" / "logs" /
    "fixed_week4_report.txt"
)

SEMANA_CORTE = 4

print("Cargando conjunto de predicción...")
df = pd.read_parquet(INPUT_FILE)

# Conservar únicamente observaciones de la semana 4
df = df[df["snapshot_week"] == SEMANA_CORTE].copy()

# El resultado final se conserva como etiqueta,
# pero nunca será una variable predictora.
features = [
    "weekly_clicks",
    "active_days",
    "activity_records",
    "has_previous_week",
    "weekly_change",
    "activity_decreased"
]

# Comprobar qué variables están disponibles
features = [col for col in features if col in df.columns]

# Eliminar filas sin etiqueta
df = df.dropna(subset=["target_withdrawn"])

# Crear conjunto final para el primer experimento
columns = [
    "code_module",
    "code_presentation",
    "id_student",
    "target_final_result",
    "target_withdrawn"
] + features

dataset = df[columns].copy()

# Resumen de la variable objetivo
summary = dataset.groupby(
    ["target_final_result", "target_withdrawn"]
).size().reset_index(name="cantidad")

# Calcular proporciones de la etiqueta binaria
total = len(dataset)
withdrawn = int(dataset["target_withdrawn"].sum())
no_withdrawn = total - withdrawn

# Guardar archivos
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

dataset.to_parquet(OUTPUT_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("CONJUNTO CON SEMANA FIJA\n")
    file.write(f"Semana de corte: {SEMANA_CORTE}\n")
    file.write(f"Total de observaciones: {total}\n")
    file.write(f"Withdrawn: {withdrawn}\n")
    file.write(f"No Withdrawn: {no_withdrawn}\n")
    file.write(f"Porcentaje Withdrawn: {withdrawn / total * 100:.2f}%\n")
    file.write("\nVariables candidatas:\n")
    file.write("\n".join(features))
    file.write(
        "\n\nLimitacion: solo incluye estudiantes con actividad "
        "registrada en la semana 4.\n"
    )
    file.write(
        "No representa necesariamente a todos los estudiantes "
        "matriculados.\n"
    )

print("\nConjunto de semana fija creado.")
print("Semana de corte:", SEMANA_CORTE)
print("Total de observaciones:", total)
print("Withdrawn:", withdrawn)
print("No Withdrawn:", no_withdrawn)
print(f"Porcentaje Withdrawn: {withdrawn / total * 100:.2f}%")

print("\nDistribución por resultado final:")
print(summary.to_string(index=False))

print("\nVariables candidatas:")
print(features)

print("\nArchivo:", OUTPUT_FILE)
print("Resumen:", SUMMARY_FILE)
print("Informe:", REPORT_FILE)
