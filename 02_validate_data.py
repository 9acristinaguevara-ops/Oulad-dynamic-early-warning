"""Comprueba presencia, columnas y lectura básica de los CSV de OULAD."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "raw" / "csv"
LOG_DIR = ROOT / "outputs" / "logs"
EXPECTED_COLUMNS = {
    "assessments.csv": {"code_module", "code_presentation", "id_assessment"},
    "courses.csv": {"code_module", "code_presentation"},
    "studentAssessment.csv": {"id_assessment", "id_student", "score"},
    "studentInfo.csv": {"code_module", "code_presentation", "id_student", "final_result"},
    "studentRegistration.csv": {"code_module", "code_presentation", "id_student"},
    "studentVle.csv": {"code_module", "code_presentation", "id_student", "date", "sum_click"},
    "vle.csv": {"id_site", "code_module", "code_presentation"},
}


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    failed = False

    for filename, required in EXPECTED_COLUMNS.items():
        path = CSV_DIR / filename
        if not path.exists():
            report.append(f"[FALTA] {filename}")
            failed = True
            continue
        try:
            # Leer solo la cabecera primero evita cargar archivos grandes innecesariamente.
            columns = set(pd.read_csv(path, nrows=0).columns)
            missing_columns = required - columns
            size_mb = path.stat().st_size / (1024 ** 2)
            report.append(f"[OK] {filename} | {size_mb:.1f} MB | columnas: {len(columns)}")
            if missing_columns:
                report.append(f"  [ERROR] Faltan columnas: {', '.join(sorted(missing_columns))}")
                failed = True
        except Exception as exc:
            report.append(f"[ERROR] {filename}: {exc}")
            failed = True

    report.append("\nResultado: " + ("REVISAR ERRORES" if failed else "CSV requeridos encontrados"))
    output = "\n".join(report)
    print(output)
    (LOG_DIR / "data_validation.txt").write_text(output, encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
