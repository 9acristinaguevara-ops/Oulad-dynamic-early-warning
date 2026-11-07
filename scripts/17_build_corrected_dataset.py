
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

INFO_FILE = (
    BASE_DIR / "data" / "raw" / "csv" /
    "studentInfo.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "interim" /
    "corrected_week4_dataset.parquet"
)

SUMMARY_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "corrected_dataset_summary.csv"
)

REPORT_FILE = (
    BASE_DIR / "outputs" / "logs" /
    "corrected_dataset_report.txt"
)

# La semana 4 comprende los dias 28 a 34,
# si week se calculo mediante date // 7.
DIA_FIN_SEMANA_4 = 34
CODIGO_SIN_RETIRADA = -1

print("Cargando actividad de la semana 4...")
activity = pd.read_parquet(DATA_FILE)

print("Cargando inscripciones...")
registration = pd.read_csv(
    REGISTRATION_FILE,
    na_values=["?"]
)

print("Cargando resultados finales...")
info = pd.read_csv(INFO_FILE)

registration["date_unregistration"] = pd.to_numeric(
    registration["date_unregistration"],
    errors="coerce"
)

df = activity.merge(
    registration,
    on=[
        "code_module",
        "code_presentation",
        "id_student"
    ],
    how="left",
    validate="one_to_one"
)

df = df.merge(
    info[
        [
            "code_module",
            "code_presentation",
            "id_student",
            "final_result"
        ]
    ],
    on=[
        "code_module",
        "code_presentation",
        "id_student"
    ],
    how="left",
    validate="one_to_one"
)

# Clasificar los casos.
df["motivo"] = "Pendiente de clasificar"

# Los resultados finales distintos de Withdrawn
# se consideran casos sin abandono.
df.loc[
    df["final_result"].isin(
        ["Pass", "Fail", "Distinction"]
    ),
    "motivo"
] = "No abandono segun resultado final"

# Abandono posterior al final de la semana 4.
df.loc[
    (df["final_result"] == "Withdrawn")
    & (df["date_unregistration"] > DIA_FIN_SEMANA_4),
    "motivo"
] = "Abandono posterior a semana 4"

# Retirada registrada hasta el final de la semana 4.
# El codigo -1 no se considera una fecha real.
df.loc[
    (df["final_result"] == "Withdrawn")
    & (df["date_unregistration"].notna())
    & (df["date_unregistration"] != CODIGO_SIN_RETIRADA)
    & (df["date_unregistration"] <= DIA_FIN_SEMANA_4),
    "motivo"
] = "Excluir retirada hasta semana 4"

# Retirada sin fecha util para determinar el momento.
df.loc[
    (df["final_result"] == "Withdrawn")
    & (
        df["date_unregistration"].isna()
        | (
            df["date_unregistration"]
            == CODIGO_SIN_RETIRADA
        )
    ),
    "motivo"
] = "Excluir fecha desconocida o codigo -1"

# Conservar solamente casos que pueden etiquetarse.
eligible = df[
    df["motivo"].isin(
        [
            "No abandono segun resultado final",
            "Abandono posterior a semana 4"
        ]
    )
].copy()

eligible["target_future_withdrawal"] = 0

eligible.loc[
    eligible["motivo"] == "Abandono posterior a semana 4",
    "target_future_withdrawal"
] = 1

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

summary = (
    df["motivo"]
    .value_counts()
    .rename_axis("categoria")
    .reset_index(name="cantidad")
)

summary.to_csv(SUMMARY_FILE, index=False)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("CONJUNTO PROSPECTIVO CORREGIDO\n\n")
    file.write(summary.to_string(index=False))

    file.write("\n\nDistribucion del objetivo:\n")
    file.write(
        eligible["target_future_withdrawal"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    file.write(
        "\n\nNota: el conjunto contiene estudiantes con "
        "actividad en la semana 4. Se consideran positivos "
        "los estudiantes con resultado Withdrawn y fecha "
        "de retirada posterior al dia 34. Los codigos -1 "
        "y las fechas desconocidas se excluyen para los "
        "casos Withdrawn.\n"
    )

print("\nRESUMEN DE CLASIFICACION")
print(summary.to_string(index=False))

print("\nDISTRIBUCION DEL OBJETIVO")
print(
    eligible["target_future_withdrawal"]
    .value_counts()
    .sort_index()
)

print("\nTotal de estudiantes elegibles:", len(eligible))
print("Archivo guardado en:", OUTPUT_FILE)
print("Resumen guardado en:", SUMMARY_FILE)
print("Informe guardado en:", REPORT_FILE)

