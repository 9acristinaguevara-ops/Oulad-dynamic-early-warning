
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

REGISTRATION_FILE = (
    BASE_DIR / "data" / "raw" / "csv" /
    "studentRegistration.csv"
)

INFO_FILE = (
    BASE_DIR / "data" / "raw" / "csv" /
    "studentInfo.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "registration_results_audit.csv"
)

REPORT_FILE = (
    BASE_DIR / "outputs" / "logs" /
    "registration_results_audit.txt"
)

print("Cargando inscripciones...")
registration = pd.read_csv(
    REGISTRATION_FILE,
    na_values=["?"]
)

print("Cargando resultados finales...")
info = pd.read_csv(INFO_FILE)

# Convertir la fecha a números.
registration["date_unregistration"] = pd.to_numeric(
    registration["date_unregistration"],
    errors="coerce"
)

# Unir inscripciones y resultados.
df = registration.merge(
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

# Clasificar las fechas sin asumir que todo negativo es -1.
df["grupo_retirada"] = "Fecha negativa distinta de -1"

df.loc[
    df["date_unregistration"].isna(),
    "grupo_retirada"
] = "Fecha desconocida"

df.loc[
    df["date_unregistration"] == -1,
    "grupo_retirada"
] = "Sin retirada registrada"

df.loc[
    df["date_unregistration"] >= 0,
    "grupo_retirada"
] = "Con retirada registrada"

print("\nValores de fecha:")
print(
    df["date_unregistration"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\nResultados finales:")
print(df["final_result"].value_counts(
    dropna=False
).to_string())

print("\nCruce entre retirada y resultado final:")

tabla = pd.crosstab(
    df["grupo_retirada"],
    df["final_result"],
    margins=True
)

print(tabla.to_string())

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

tabla.to_csv(OUTPUT_FILE)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("AUDITORIA DE INSCRIPCIONES Y RESULTADOS\n\n")
    file.write(tabla.to_string())
    file.write("\n\nCantidad de fechas desconocidas: ")
    file.write(str(df["date_unregistration"].isna().sum()))
    file.write("\n")

print("\nTabla guardada en:", OUTPUT_FILE)
print("Informe guardado en:", REPORT_FILE)


