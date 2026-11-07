
import pandas as pd

archivo = "data/interim/corrected_week4_dataset.parquet"

data = pd.read_parquet(archivo)

print("\nPROMEDIOS SEGUN RETIRADA FUTURA")

resumen = data.groupby(
    "target_future_withdrawal"
)[
    [
        "weekly_clicks",
        "active_days",
        "activity_records",
        "weekly_change",
        "activity_decreased"
    ]
].mean()

print(resumen.round(2))

print("\nMEDIANAS SEGUN RETIRADA FUTURA")

medianas = data.groupby(
    "target_future_withdrawal"
)[
    [
        "weekly_clicks",
        "active_days",
        "activity_records",
        "weekly_change"
    ]
].median()

print(medianas)

print("\nCANTIDAD DE ESTUDIANTES POR GRUPO")

print(data["target_future_withdrawal"].value_counts())
