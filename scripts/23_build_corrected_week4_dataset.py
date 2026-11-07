
import pandas as pd
from pathlib import Path

# Rutas del proyecto
BASE = Path(".")
RAW = BASE / "data" / "raw" / "csv"
INTERIM = BASE / "data" / "interim"
TABLES = BASE / "outputs" / "tables"
LOGS = BASE / "outputs" / "logs"

# Leer los archivos
actividad = pd.read_parquet(
    INTERIM / "weekly_activity.parquet"
)

registro = pd.read_csv(
    RAW / "studentRegistration.csv",
    na_values=["?"]
)

estudiantes = pd.read_csv(
    RAW / "studentInfo.csv"
)

# Convertir fechas a valores numéricos
registro["date_registration"] = pd.to_numeric(
    registro["date_registration"], errors="coerce"
)

registro["date_unregistration"] = pd.to_numeric(
    registro["date_unregistration"], errors="coerce"
)

# Unir información de matrícula y resultado final
claves = [
    "code_module",
    "code_presentation",
    "id_student"
]

datos = registro.merge(
    estudiantes[claves + ["final_result"]],
    on=claves,
    how="left",
    validate="one_to_one"
)

# Obtener actividad de la semana 3
semana3 = actividad[actividad["week"] == 3][
    claves + ["weekly_clicks"]
].copy()

semana3 = semana3.rename(
    columns={"weekly_clicks": "week3_clicks"}
)

# Obtener actividad de la semana 4
semana4 = actividad[actividad["week"] == 4][
    claves + [
        "weekly_clicks",
        "active_days",
        "activity_records"
    ]
].copy()

semana4 = semana4.rename(columns={
    "weekly_clicks": "week4_clicks",
    "active_days": "week4_active_days",
    "activity_records": "week4_activity_records"
})

# Añadir las dos semanas a todas las matrículas
datos = datos.merge(
    semana3, on=claves, how="left", validate="one_to_one"
)

datos = datos.merge(
    semana4, on=claves, how="left", validate="one_to_one"
)

# Determinar si existe actividad registrada en semana 4
datos["has_week4_activity"] = (
    datos["week4_activity_records"].notna().astype(int)
)

# Calcular cambios de actividad
datos["weekly_change"] = (
    datos["week4_clicks"] - datos["week3_clicks"]
)

# No calcular cambio si falta actividad en alguna semana
datos.loc[
    datos["week3_clicks"].isna()
    | datos["week4_clicks"].isna(),
    "weekly_change"
] = pd.NA

# Identificar matrículas válidas para una alerta al día 34
matricula_valida = (
    datos["date_registration"].notna()
    & (datos["date_registration"] <= 34)
)

# Identificar estudiantes que ya se habían retirado al día 34
retirada_temprana = (
    datos["date_unregistration"].notna()
    & (datos["date_unregistration"] <= 34)
)

# Crear etiquetas a partir de las fechas
datos["target_future_withdrawal"] = pd.NA

# Retirada después del día 34
retirada_futura = (
    datos["date_unregistration"].notna()
    & (datos["date_unregistration"] > 34)
)

datos.loc[retirada_futura, "target_future_withdrawal"] = 1

# Resultados sin retirada futura y sin fecha de retirada
sin_retirada = (
    datos["final_result"].isin(["Pass", "Fail", "Distinction"])
    & datos["date_unregistration"].isna()
)

datos.loc[sin_retirada, "target_future_withdrawal"] = 0

# Excluir matrículas no válidas y retiradas tempranas
elegibles = datos[
    matricula_valida
    & ~retirada_temprana
    & datos["target_future_withdrawal"].notna()
].copy()

elegibles["target_future_withdrawal"] = (
    elegibles["target_future_withdrawal"].astype(int)
)

# Guardar el conjunto nuevo
INTERIM.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

elegibles.to_parquet(
    INTERIM / "corrected_week4_final_dataset.parquet",
    index=False
)

# Resumen por etiqueta y actividad
resumen = elegibles.groupby(
    ["target_future_withdrawal", "has_week4_activity"]
).size().reset_index(name="cantidad")

resumen.to_csv(
    TABLES / "corrected_week4_final_summary.csv",
    index=False
)

# Reporte
with open(
    LOGS / "corrected_week4_final_report.txt",
    "w",
    encoding="utf-8"
) as archivo:
    archivo.write("CONJUNTO CORREGIDO DE SEMANA 4\n")
    archivo.write(f"Total de matrículas elegibles: {len(elegibles)}\n")
    archivo.write(
        f"Retiradas futuras: "
        f"{(elegibles['target_future_withdrawal'] == 1).sum()}\n"
    )
    archivo.write(
        f"Sin retirada futura: "
        f"{(elegibles['target_future_withdrawal'] == 0).sum()}\n"
    )
    archivo.write(
        f"Sin actividad registrada en semana 4: "
        f"{(elegibles['has_week4_activity'] == 0).sum()}\n"
    )

print("CONJUNTO CORREGIDO CREADO")
print("Total elegible:", len(elegibles))
print("\nDistribución de etiquetas:")
print(elegibles["target_future_withdrawal"].value_counts().sort_index())
print("\nActividad semana 4 por etiqueta:")
print(pd.crosstab(
    elegibles["target_future_withdrawal"],
    elegibles["has_week4_activity"]
).to_string())
print("\nArchivos guardados:")
print(INTERIM / "corrected_week4_final_dataset.parquet")
print(TABLES / "corrected_week4_final_summary.csv")
print(LOGS / "corrected_week4_final_report.txt")
