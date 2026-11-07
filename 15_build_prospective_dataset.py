
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
    BASE_DIR / "data" / "interim" /
    "prospective_week4_dataset.parquet"
)

SUMMARY_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "prospective_dataset_summary.csv"
)

REPORT_FILE = (
    BASE_DIR / "outputs" / "logs" /
    "prospective_dataset_report.txt"
)

SEMANA_CORTE = 4

print("Cargando datos de la semana 4...")
df = pd.read_parquet(DATA_FILE)

print("Cargando datos de inscripción...")
registration = pd.read_csv(REGISTRATION_FILE)

registration = registration[
    [
        "code_module",
        "code_presentation",
        "id_student",
        "date_registration",
        "date_unregistration"
    ]
].copy()

registration["date_registration"] = pd.to_numeric(
    registration["date_registration"],
    errors="coerce"
)

registration["date_unregistration"] = pd.to_numeric(
    registration["date_unregistration"],
    errors="coerce"
)

df = df.merge(
    registration,
    on=[
        "code_module",
        "code_presentation",
        "id_student"
    ],
    how="left",
    validate="one_to_one"
)

# Identificar estudiantes retirados antes o durante la semana 4.
df["exclusion_reason"] = ""

df.loc[
    df["date_unregistration"].notna()
    & (df["date_unregistration"] >= 0)
    & (df["date_unregistration"] <= SEMANA_CORTE),
    "exclusion_reason"
] = "Retirada antes o durante semana 4"

# Identificar registros sin información de inscripción.
df.loc[
    df["date_unregistration"].isna(),
    "exclusion_reason"
] = "Fecha de retirada desconocida"

# Conservar estudiantes elegibles para la predicción.
eligible = df[
    df["exclusion_reason"] == ""
].copy()

# Objetivo: retirada posterior a la semana 4.
eligible["target_future_withdrawal"] = (
    eligible["date_unregistration"] > SEMANA_CORTE
).astype(int)

# Preparar las variables predictoras.
features = [
    "code_module",
    "code_presentation",
    "id_student",
    "weekly_clicks",
    "active_days",
    "activity_records",
    "has_previous_week",
    "weekly_change",
    "activity_decreased",
    "target_future_withdrawal"
]

eligible = eligible[features].copy()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

eligible.to_parquet(OUTPUT_FILE, index=False)

# Resumen de los registros originales y elegibles.
summary = pd.DataFrame({
    "categoria": [
        "Registros iniciales",
        "Retiradas antes o durante semana 4",
        "Fecha de retirada desconocida",
        "Estudiantes elegibles",
        "Retirada posterior a semana 4",
        "Sin retirada posterior registrada"
    ],
    "cantidad": [
        len(df),
        (df["exclusion_reason"] ==
         "Retirada antes o durante semana 4").sum(),
        (df["exclusion_reason"] ==
         "Fecha de retirada desconocida").sum(),
        len(eligible),
        (eligible["target_future_withdrawal"] == 1).sum(),
        (eligible["target_future_withdrawal"] == 0).sum()
    ]
})

summary.to_csv(SUMMARY_FILE, index=False)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("CONJUNTO DE PREDICCION PROSPECTIVA\n\n")
    file.write(summary.to_string(index=False))
    file.write("\n\nDistribucion del objetivo:\n")
    file.write(
        eligible["target_future_withdrawal"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    file.write(
        "\n\nLimitacion: el conjunto inicial contiene solamente "
        "estudiantes con actividad registrada en la semana 4. "
        "Por ello, no representa necesariamente a todos los "
        "estudiantes inscritos en ese momento.\n"
    )

print("\nRESUMEN DEL CONJUNTO PROSPECTIVO")
print(summary.to_string(index=False))

print("\nDistribución del objetivo:")
print(
    eligible["target_future_withdrawal"]
    .value_counts()
    .sort_index()
)

print("\nArchivo guardado en:", OUTPUT_FILE)
print("Resumen guardado en:", SUMMARY_FILE)
print("Informe guardado en:", REPORT_FILE)
