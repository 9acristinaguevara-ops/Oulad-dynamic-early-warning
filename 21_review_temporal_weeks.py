
import pandas as pd

archivo = "data/interim/temporal_student_activity.parquet"

data = pd.read_parquet(archivo)

print("\nDIMENSIONES")
print(data.shape)

print("\nCOLUMNAS")
print(data.columns.tolist())

print("\nSEMANAS DISPONIBLES")
print(sorted(data["week"].unique()))

print("\nPRIMERAS FILAS")
print(data.head(10).to_string(index=False))

print("\nCANTIDAD DE REGISTROS POR SEMANA")
print(data["week"].value_counts().sort_index())
