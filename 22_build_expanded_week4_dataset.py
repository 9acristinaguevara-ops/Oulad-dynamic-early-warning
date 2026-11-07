
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ACTIVITY_FILE = (
    BASE_DIR / "data" / "interim" /
    "weekly_activity.parquet"
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
    "expanded_week4_dataset.parquet"
)

SUMMARY_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "expanded_week4_summary.csv"
)

REPORT_FILE = (
    BASE_DIR / "outputs" / "logs" /
    "expanded_week4_report.txt"
)

SEMANA_CORTE = 4
DIA_CORTE = 34

print("Cargando actividad semanal...")
activity = pd.read_parquet(ACTIVITY_FILE)

print("Cargando matrículas...")
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

keys = [
    "code_module",
    "code_presentation",
    "id_student"
]

# Crear la base con todas las matrículas.
df = registration.merge(
    info[keys + ["final_result"]],
    on=keys,
    how="inner",
    validate="one_to_one"
)

# Incorporar la actividad de la semana 4.
week4 = activity[
    activity["week"] == SEMANA_CORTE
].copy()

week4 = week4[
    keys + [
        "weekly_clicks",
        "active_days",
        "activity_records"
    ]
]

week4 = week4.rename(
    columns={
        "weekly_clicks": "week4_clicks",
        "active_days": "week4_active_days",
        "activity_records": "week4_activity_records"
    }
)

df = df.merge(
    week4,
    on=keys,
    how="left",
    validate="one_to_one"
)

df["has_week4_activity"] = (
    df["week4_activity_records"].notna().astype(int)
)

# Incorporar la actividad de la semana 3.
week3 = activity[
    activity["week"] == SEMANA_CORTE - 1
][
    keys + ["weekly_clicks"]
].copy()

week3 = week3.rename(
    columns={"weekly_clicks": "week3_clicks"}
)

df = df.merge(
    week3,
    on=keys,
    how="left",
    validate="one_to_one"
)

# Crear la etiqueta y clasificar los casos.
df["target_future_withdrawal"] = pd.NA
df["motivo"] = "Excluir"

# Resultados finales que no indican retirada.
no_withdrawal = df["final_result"].isin(
    ["Pass", "Fail", "Distinction"]
)

df.loc[
    no_withdrawal,
    "target_future_withdrawal"
] = 0

df.loc[
    no_withdrawal,
    "motivo"
] = "Resultado final sin retirada"

# Retirada posterior al corte.
future_withdrawal = (
    (df["final_result"] == "Withdrawn")
    & (df["date_unregistration"] > DIA_CORTE)
)

df.loc[
    future_withdrawal,
    "target_future_withdrawal"
] = 1

df.loc[
    future_withdrawal,
    "motivo"
] = "Retirada posterior al corte"

# Excluir estudiantes cuya retirada ya ocurrió
# hasta el día 34, o cuya fecha no se conoce.
eligible = df[
    df["target_future_withdrawal"].notna()
].copy()

eligible["target_future_withdrawal"] = (
    eligible["target_future_withdrawal"].astype(int)
)

# Diferencia de actividad solo si se conocen ambas semanas.
eligible["weekly_change"] = (
    eligible["week4_clicks"] - eligible["week3_clicks"]
)

eligible.loc[
    eligible["has_week4_activity"] == 0,
    "weekly_change"
] = pd.NA

# Guardar variables del conjunto experimental.
columns = keys + [
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "week3_clicks",
    "weekly_change",
    "target_future_withdrawal",
    "motivo"
]

dataset = eligible[columns].copy()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

dataset.to_parquet(OUTPUT_FILE, index=False)

summary = (
    dataset.groupby(
        ["motivo", "target_future_withdrawal"]
    )
    .size()
    .reset_index(name="cantidad")
)

summary.to_csv(SUMMARY_FILE, index=False)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("CONJUNTO EXPANDIDO DE SEMANA 4\n\n")
    file.write(f"Total elegible: {len(dataset)}\n")
    file.write(
        "Con actividad semana 4: "
        f"{dataset['has_week4_activity'].sum()}\n"
    )
    file.write(
        "Sin actividad semana 4: "
        f"{(dataset['has_week4_activity'] == 0).sum()}\n"
    )
    file.write("\nDistribución de etiquetas:\n")
    file.write(
        dataset["target_future_withdrawal"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    file.write("\n\nResumen de casos:\n")
    file.write(summary.to_string(index=False))
    file.write(
        "\n\nNota: la ausencia de registros de actividad "
        "no se interpreta automáticamente como cero clics."
    )

print("\nCONJUNTO EXPANDIDO CREADO")
print("Total elegible:", len(dataset))
print(
    "Con actividad semana 4:",
    int(dataset["has_week4_activity"].sum())
)
print(
    "Sin actividad semana 4:",
    int((dataset["has_week4_activity"] == 0).sum())
)
print("\nDistribución del objetivo:")
print(dataset["target_future_withdrawal"].value_counts().sort_index())
print("\nResumen:")
print(summary.to_string(index=False))
print("\nArchivo guardado en:", OUTPUT_FILE)
print("Informe guardado en:", REPORT_FILE)
