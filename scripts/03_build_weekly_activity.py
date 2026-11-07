"""Construye actividad semanal a partir de studentVle.csv.

Esta primera versión resume clics observados. No estima fatiga, no define aún
el objetivo de abandono y no ajusta ecuaciones diferenciales.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "csv" / "studentVle.csv"
OUTPUT_DIR = ROOT / "data" / "interim"
OUTPUT = OUTPUT_DIR / "weekly_activity.parquet"
CSV_OUTPUT = OUTPUT_DIR / "weekly_activity.csv"


def main():
    if not INPUT.exists():
        print(f"No se encuentra {INPUT}")
        print(r"Primero ejecuta: python scripts\01_download_oulad.py")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["code_module", "code_presentation", "id_student", "date", "sum_click"]
    print("Leyendo studentVle.csv; el archivo puede tardar un poco...")
    data = pd.read_csv(INPUT, usecols=columns)

    # OULAD registra días relativos al inicio de la presentación.
    # La semana se calcula con división entera. Los valores negativos de date,
    # si existen, se conservan en semanas negativas para inspección posterior.
    data["week"] = (data["date"] // 7).astype("int32")

    group_columns = [
        "code_module", "code_presentation", "id_student", "week"
    ]
    weekly = (
        data.groupby(group_columns, as_index=False)
        .agg(
            weekly_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            activity_records=("sum_click", "size"),
        )
    )

    weekly.to_parquet(OUTPUT, index=False)
    # CSV adicional para inspección con herramientas sencillas; puede ser grande.
    weekly.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
    print(f"Filas de entrada: {len(data):,}")
    print(f"Filas agregadas: {len(weekly):,}")
    print(f"Parquet guardado en: {OUTPUT}")
    print(f"CSV guardado en: {CSV_OUTPUT}")
    print("Siguiente paso: inspeccionar la cobertura temporal y unir con registros/resultados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
