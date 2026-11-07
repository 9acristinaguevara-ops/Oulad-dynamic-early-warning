
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "interim" /
    "prediction_snapshot_week4.parquet"
)

OUTPUT_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "prediction_feature_review.csv"
)

print("Cargando conjunto de predicción...")
df = pd.read_parquet(INPUT_FILE)

print("\n1. DIMENSIONES DEL CONJUNTO")
print("Filas:", len(df))
print("Columnas:", len(df.columns))

print("\n2. COLUMNAS DISPONIBLES")
print(df.columns.tolist())

print("\n3. TIPO DE DATO Y VALORES AUSENTES")

resumen = pd.DataFrame({
    "columna": df.columns,
    "tipo_dato": [str(df[col].dtype) for col in df.columns],
    "valores_ausentes": [df[col].isna().sum() for col in df.columns],
    "porcentaje_ausente": [
        round(df[col].isna().mean() * 100, 2)
        for col in df.columns
    ],
    "valores_distintos": [df[col].nunique() for col in df.columns]
})

print(resumen.to_string(index=False))

print("\n4. DISTRIBUCIÓN POR SEMANA DE CORTE")
print(
    df["snapshot_week"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n5. ACTIVIDAD SEGÚN EL RESULTADO FINAL")

print(
    df.groupby("target_final_result").agg(
        casos=("id_student", "count"),
        semana_media=("snapshot_week", "mean"),
        clics_medios=("weekly_clicks", "mean"),
        dias_activos_medios=("active_days", "mean"),
        registros_medios=("activity_records", "mean")
    ).round(2).to_string()
)

print("\n6. REVISIÓN DE VARIABLES SOSPECHOSAS")

columnas_no_predictoras = [
    "id_student",
    "code_module",
    "code_presentation",
    "target_final_result",
    "target_withdrawn",
    "snapshot_week"
]

print("Columnas que no deben entrar directamente al modelo:")
for col in columnas_no_predictoras:
    if col in df.columns:
        print("-", col)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
resumen.to_csv(OUTPUT_FILE, index=False)

print("\nRevisión terminada.")
print("Informe guardado en:", OUTPUT_FILE)
