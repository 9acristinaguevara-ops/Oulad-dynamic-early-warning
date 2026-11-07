
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. RUTAS DEL PROYECTO
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "interim"
    / "temporal_student_activity.parquet"
)

TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
LOGS_DIR = BASE_DIR / "outputs" / "logs"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CARGAR LOS DATOS
# ==========================================

print("Cargando conjunto temporal...")
data = pd.read_parquet(INPUT_FILE)

keys = [
    "code_module",
    "code_presentation",
    "id_student"
]

data = data.sort_values(
    keys + ["week"]
).reset_index(drop=True)

print("Registros cargados:", len(data))


# ==========================================
# 3. CALCULAR CAMBIOS SEMANALES
# ==========================================

data["previous_clicks"] = (
    data.groupby(keys)["weekly_clicks"].shift(1)
)

previous_week = data.groupby(keys)["week"].shift(1)

data["consecutive"] = (
    data["week"] == previous_week + 1
)

# Diferencia de clics entre semanas consecutivas
data["weekly_difference"] = (
    data["weekly_clicks"] - data["previous_clicks"]
)

data.loc[
    ~data["consecutive"],
    "weekly_difference"
] = pd.NA

# Descenso relativo: valores positivos indican reducción
data["activity_decline"] = pd.NA

valid = (
    data["consecutive"]
    & data["previous_clicks"].notna()
    & (data["previous_clicks"] > 0)
)

data.loc[valid, "activity_decline"] = (
    (
        data.loc[valid, "previous_clicks"]
        - data.loc[valid, "weekly_clicks"]
    )
    / data.loc[valid, "previous_clicks"]
)


# ==========================================
# 4. RESUMEN POR ESTUDIANTE Y CURSO
# ==========================================

student_summary = (
    data.groupby(keys, as_index=False)
    .agg(
        observed_weeks=("week", "nunique"),
        mean_clicks=("weekly_clicks", "mean"),
        median_clicks=("weekly_clicks", "median"),
        std_clicks=("weekly_clicks", "std"),
        total_clicks=("weekly_clicks", "sum"),
        mean_active_days=("active_days", "mean"),
        final_result=("final_result", "first")
    )
)

# Variabilidad relativa de la actividad
student_summary["relative_variability"] = (
    student_summary["std_clicks"]
    / student_summary["mean_clicks"]
)

# Resumen de descensos por estudiante y curso
decline_summary = (
    data.groupby(keys, as_index=False)
    .agg(
        mean_decline=("activity_decline", "mean"),
        maximum_decline=("activity_decline", "max"),
        valid_decline_weeks=("activity_decline", "count")
    )
)

student_summary = student_summary.merge(
    decline_summary,
    on=keys,
    how="left",
    validate="one_to_one"
)

student_summary.to_csv(
    TABLES_DIR / "student_activity_dynamics.csv",
    index=False
)


# ==========================================
# 5. COMPARACIÓN POR RESULTADO FINAL
# ==========================================

result_summary = (
    student_summary.groupby(
        "final_result",
        dropna=False,
        as_index=False
    )
    .agg(
        student_courses=("id_student", "size"),
        median_mean_clicks=("mean_clicks", "median"),
        median_active_days=("mean_active_days", "median"),
        median_variability=("relative_variability", "median"),
        median_decline=("mean_decline", "median")
    )
)

result_summary.to_csv(
    TABLES_DIR / "dynamics_by_final_result.csv",
    index=False
)

print("\nComparación descriptiva por resultado:")
print(result_summary.to_string(index=False))


# ==========================================
# 6. GRÁFICO DE ACTIVIDAD POR RESULTADO
# ==========================================

weekly_result = (
    data.groupby(
        ["week", "final_result"],
        as_index=False,
        dropna=False
    )
    .agg(
        median_clicks=("weekly_clicks", "median")
    )
)

plt.figure(figsize=(10, 6))

for result, group in weekly_result.groupby("final_result"):
    plt.plot(
        group["week"],
        group["median_clicks"],
        label=str(result)
    )

plt.xlabel("Semana del curso")
plt.ylabel("Mediana de clics")
plt.title("Actividad semanal según resultado final")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "04_dynamics_by_final_result.png",
    dpi=300
)

plt.close()


# ==========================================
# 7. GRÁFICO DE VARIABILIDAD
# ==========================================

plot_data = student_summary.dropna(
    subset=["mean_clicks", "relative_variability"]
)

plt.figure(figsize=(9, 6))

plt.scatter(
    plot_data["mean_clicks"],
    plot_data["relative_variability"],
    alpha=0.35
)

plt.xlabel("Promedio de clics")
plt.ylabel("Variabilidad relativa")
plt.title("Variabilidad de actividad por estudiante-curso")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "05_activity_variability.png",
    dpi=300
)

plt.close()


# ==========================================
# 8. GUARDAR INFORME
# ==========================================

with open(
    LOGS_DIR / "activity_dynamics_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("ANALISIS DE DINAMICA DE ACTIVIDAD\n")
    file.write("=================================\n\n")

    file.write(f"Registros analizados: {len(data)}\n")
    file.write(
        f"Estudiantes-curso analizados: {len(student_summary)}\n\n"
    )

    file.write("COMPARACION POR RESULTADO FINAL\n")
    file.write(result_summary.to_string(index=False))
    file.write("\n\n")

    file.write(
        "Este analisis es descriptivo y no demuestra causalidad.\n"
    )

    file.write(
        "Todavia no se ha validado un sistema de prediccion temprana.\n"
    )


print("\nAnálisis de dinámica terminado.")
print(
    "Tabla de estudiantes:",
    TABLES_DIR / "student_activity_dynamics.csv"
)
print(
    "Tabla por resultado:",
    TABLES_DIR / "dynamics_by_final_result.csv"
)
print("Gráficos guardados en:", FIGURES_DIR)
print("Informe guardado en:", LOGS_DIR)
