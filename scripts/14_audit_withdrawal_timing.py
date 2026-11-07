
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "data" / "interim" /
    "fixed_week4_dataset.parquet"
)

REGISTRATION_FILE = (
    BASE_DIR / "data" / "raw" / "csv" /
    "studentRegistration.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "withdrawal_timing_audit.csv"
)

REPORT_FILE = (
    BASE_DIR / "outputs" / "logs" /
    "withdrawal_timing_audit.txt"
)

SEMANA_CORTE = 4

print("Cargando conjunto de la semana 4...")
df = pd.read_parquet(DATA_FILE)

print("Cargando fechas de registro...")
registration = pd.read_csv(REGISTRATION_FILE)

print("\nColumnas del archivo de registro:")
print(registration.columns.tolist())

# Conservar solo las columnas necesarias
registration = registration[
    [
        "code_module",
        "code_presentation",
        "id_student",
        "date_registration",
        "date_unregistration"
    ]
].copy()

# Convertir las fechas relativas a números
registration["date_registration"] = pd.to_numeric(
    registration["date_registration"],
    errors="coerce"
)

registration["date_unregistration"] = pd.to_numeric(
    registration["date_unregistration"],
    errors="coerce"
)

# Unir usando las claves del estudiante y del curso
df = df.merge(
    registration,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
    validate="many_to_one"
)

print("\nRegistros sin fecha de retirada disponible:")
print(df["date_unregistration"].isna().sum())

# En OULAD, -1 en date_unregistration significa
# que no hay una fecha de retirada registrada.
df["withdrawal_status"] = "Sin fecha disponible"

df.loc[
    df["date_unregistration"] == -1,
    "withdrawal_status"
] = "Sin retirada registrada"

df.loc[
    df["date_unregistration"] >= 0,
    "withdrawal_status"
] = "Retirada con fecha"

# Clasificar las retiradas con fecha respecto a la semana 4
df["withdrawal_timing"] = "No determinado"

df.loc[
    df["date_unregistration"] == -1,
    "withdrawal_timing"
] = "Sin retirada registrada"

df.loc[
    df["date_unregistration"] >= 0,
    "withdrawal_timing"
] = "Retirada en semana 4 o antes"

df.loc[
    df["date_unregistration"] > SEMANA_CORTE,
    "withdrawal_timing"
] = "Retirada después de semana 4"

# Resumen para estudiantes con resultado final Withdrawn
withdrawn = df[df["target_withdrawn"] == 1].copy()

summary = withdrawn.groupby(
    "withdrawal_timing",
    dropna=False
).size().reset_index(name="cantidad")

summary = summary.sort_values(
    "cantidad",
    ascending=False
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

summary.to_csv(OUTPUT_FILE, index=False)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("AUDITORIA TEMPORAL DE RETIRADAS\n")
    file.write(f"Semana de corte: {SEMANA_CORTE}\n")
    file.write(f"Casos Withdrawn revisados: {len(withdrawn)}\n\n")
    file.write(summary.to_string(index=False))
    file.write(
        "\n\nNota: date_unregistration es una semana relativa "
        "al inicio del curso. Se debe comprobar que la retirada "
        "ocurra después del punto de alerta para definir un "
        "objetivo de predicción futura.\n"
    )

print("\nDISTRIBUCIÓN DE RETIRADAS ENTRE LOS CASOS WITHDRAWN")
print(summary.to_string(index=False))

print("\nTotal de casos Withdrawn revisados:", len(withdrawn))
print("Tabla guardada en:", OUTPUT_FILE)
print("Informe guardado en:", REPORT_FILE)
