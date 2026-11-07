
"""
Script 04: análisis exploratorio de la actividad semanal de OULAD.

Genera tablas descriptivas y gráficos iniciales.
Todavía no estima fatiga ni ajusta ecuaciones diferenciales.
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Definir las rutas del proyecto
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

ACTIVITY_FILE = ROOT / "data" / "interim" / "weekly_activity.parquet"
STUDENT_FILE = ROOT / "data" / "raw" / "csv" / "studentInfo.csv"

TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"


def main():

    # Crear las carpetas para guardar los resultados.
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Comprobar que existen los archivos necesarios.
    if not ACTIVITY_FILE.exists():
        print("Error: no existe la tabla de actividad semanal.")
        print("Ejecuta primero el script 03.")
        return 1

    if not STUDENT_FILE.exists():
        print("Error: no se encuentra studentInfo.csv.")
        return 1

    # --------------------------------------------------
    # 2. Cargar los datos
    # --------------------------------------------------

    print("Cargando actividad semanal...")
    activity = pd.read_parquet(ACTIVITY_FILE)

    print("Cargando información de estudiantes...")
    students = pd.read_csv(STUDENT_FILE)

    print("Filas de actividad:", len(activity))
    print("Estudiantes únicos en actividad:",
          activity["id_student"].nunique())

    # Seleccionar únicamente las columnas que necesitamos.
    student_columns = [
        "code_module",
        "code_presentation",
        "id_student",
        "final_result"
    ]

    students = students[student_columns].copy()

    # Verificar que la clave del estudiante sea única
    # dentro de cada módulo y presentación.
    keys = [
        "code_module",
        "code_presentation",
        "id_student"
    ]

    duplicate_students = students.duplicated(
        subset=keys
    ).sum()

    if duplicate_students > 0:
        print("Error: existen registros duplicados en studentInfo.")
        print("Duplicados encontrados:", duplicate_students)
        print("Se detiene el análisis para revisar las claves.")
        return 1

    # --------------------------------------------------
    # 3. Unir actividad y resultados finales
    # --------------------------------------------------

    print("Relacionando actividad con resultados finales...")

    data = activity.merge(
        students,
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=True
    )

    # Comprobar si todos los registros de actividad
    # encontraron un estudiante correspondiente.
    unmatched = (data["_merge"] != "both").sum()

    print("Registros sin correspondencia:", unmatched)

    data = data.drop(columns=["_merge"])

    # --------------------------------------------------
    # 4. Revisar la calidad de los datos
    # --------------------------------------------------

    duplicate_weeks = data.duplicated(
        subset=keys + ["week"]
    ).sum()

    missing_results = data["final_result"].isna().sum()

    report = []

    report.append("ANÁLISIS EXPLORATORIO DE OULAD")
    report.append("=" * 45)
    report.append(f"Filas de actividad semanal: {len(data)}")
    report.append(
        f"Estudiantes únicos: {data['id_student'].nunique()}"
    )
    report.append(
        f"Módulos: {data['code_module'].nunique()}"
    )
    report.append(
        f"Presentaciones: "
        f"{data[['code_module', 'code_presentation']].drop_duplicates().shape[0]}"
    )
    report.append(
        f"Semanas mínima y máxima: "
        f"{data['week'].min()} a {data['week'].max()}"
    )
    report.append(
        f"Duplicados estudiante-módulo-presentación-semana: "
        f"{duplicate_weeks}"
    )
    report.append(
        f"Registros sin resultado final asociado: {missing_results}"
    )
    report.append(
        f"Registros sin correspondencia en studentInfo: {unmatched}"
    )

    # Resumen de las variables numéricas.
    report.append("\nESTADÍSTICAS DESCRIPTIVAS")
    report.append(str(
        data[
            [
                "weekly_clicks",
                "active_days",
                "activity_records"
            ]
        ].describe().round(2)
    ))

    # Distribución de resultados finales.
    report.append("\nRESULTADOS FINALES")
    report.append(str(
        students["final_result"].value_counts(dropna=False)
    ))

    report_text = "\n".join(report)

    print("\n" + report_text)

    (LOG_DIR / "exploratory_report.txt").write_text(
        report_text,
        encoding="utf-8"
    )

    # Guardar la distribución de resultados.
    result_counts = (
        students["final_result"]
        .value_counts(dropna=False)
        .rename_axis("final_result")
        .reset_index(name="students")
    )

    result_counts.to_csv(
        TABLE_DIR / "student_final_results.csv",
        index=False
    )

    # --------------------------------------------------
    # 5. Cobertura temporal por módulo y presentación
    # --------------------------------------------------

    coverage = (
        data.groupby(
            ["code_module", "code_presentation"],
            as_index=False
        )
        .agg(
            first_week=("week", "min"),
            last_week=("week", "max"),
            observed_weeks=("week", "nunique"),
            students=("id_student", "nunique"),
            total_clicks=("weekly_clicks", "sum")
        )
    )

    coverage.to_csv(
        TABLE_DIR / "temporal_coverage.csv",
        index=False
    )

    # --------------------------------------------------
    # 6. Actividad semanal general
    # --------------------------------------------------

    weekly_summary = (
        data.groupby("week", as_index=False)
        .agg(
            active_students=("id_student", "nunique"),
            total_clicks=("weekly_clicks", "sum"),
            median_clicks=("weekly_clicks", "median")
        )
        .sort_values("week")
    )

    weekly_summary.to_csv(
        TABLE_DIR / "weekly_activity_summary.csv",
        index=False
    )

    # Gráfico 1: cantidad total de clics por semana.
    plt.figure(figsize=(10, 5))

    plt.plot(
        weekly_summary["week"],
        weekly_summary["total_clicks"]
    )

    plt.xlabel("Semana relativa al inicio")
    plt.ylabel("Total de clics observados")
    plt.title("Actividad semanal registrada en OULAD")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "01_weekly_total_clicks.png",
        dpi=300
    )

    plt.close()

    # Gráfico 2: estudiantes activos por semana.
    plt.figure(figsize=(10, 5))

    plt.plot(
        weekly_summary["week"],
        weekly_summary["active_students"]
    )

    plt.xlabel("Semana relativa al inicio")
    plt.ylabel("Estudiantes con actividad registrada")
    plt.title("Estudiantes activos por semana en OULAD")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "02_weekly_active_students.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 7. Comparar actividad según el resultado final
    # --------------------------------------------------

    by_result = (
        data.groupby(
            ["week", "final_result"],
            as_index=False,
            dropna=False
        )
        .agg(
            median_clicks=("weekly_clicks", "median"),
            students=("id_student", "nunique")
        )
    )

    by_result.to_csv(
        TABLE_DIR / "weekly_activity_by_result.csv",
        index=False
    )


    # Crear una figura con un periodo común de observación.
    # Se excluyen las semanas posteriores a la 33 porque
    # el grupo Withdrawn tiene muy pocos registros.

    MIN_STUDENTS = 30
    MAX_WEEK = 33

    plot_data = by_result[
        (by_result["week"] <= MAX_WEEK)
        & (by_result["students"] >= MIN_STUDENTS)
    ].copy()

    plt.figure(figsize=(11, 6))

    for result in plot_data["final_result"].dropna().unique():

        subset = plot_data[
            plot_data["final_result"] == result
        ].sort_values("week")

        plt.plot(
            subset["week"],
            subset["median_clicks"],
            linewidth=2,
            label=result
        )

    plt.xlabel("Week relative to course start")
    plt.ylabel("Median weekly clicks per student-course record")
    plt.title("Weekly activity by final academic outcome")

    plt.xlim(-4, MAX_WEEK)
    plt.legend(title="Final result")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "03_activity_by_final_result.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print("\nAnálisis exploratorio terminado.")
    print("Informe:", LOG_DIR / "exploratory_report.txt")
    print("Tablas:", TABLE_DIR)
    print("Gráficos:", FIGURE_DIR)

    print(
        "\nImportante: las semanas sin registros no se han "
        "convertido todavía en semanas con cero clics."
    )

    print(
        "Los gráficos son descriptivos y no demuestran "
        "causalidad ni transiciones críticas."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
