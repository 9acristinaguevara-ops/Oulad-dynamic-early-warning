"""
Auditoria complementaria integral OULAD (experimento 142).

No modifica datos originales, datasets intermedios ni resultados de experimentos.
Reconstruye el objetivo y el panel de semanas 0-4, y compara ambos con las
versiones existentes. Todos los informes se guardan exclusivamente en
outputs/audit_142/.

Ejecutar desde la raíz del repositorio:
    python scripts/142_auditoria_complementaria.py
"""

from pathlib import Path
import platform
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "csv"
INTERIM = BASE / "data" / "interim"
OUT = BASE / "outputs" / "audit_142"

KEYS = ["code_module", "code_presentation", "id_student"]
WEEKS = list(range(5))
CLICK_COLS = [f"clics_semana_{i}" for i in WEEKS]
AVAIL_COLS = [f"actividad_registrada_semana_{i}" for i in WEEKS]
PANEL_COLS = CLICK_COLS + AVAIL_COLS


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas requeridas: {missing}")


def add_check(rows, name, value, status, detail=""):
    rows.append({
        "control": name,
        "valor": value,
        "estado": status,
        "detalle": detail
    })


def compare_frames(left, right, columns, suffixes=("_reconstruido", "_existente")):
    return left[KEYS + columns].merge(
        right[KEYS + columns],
        on=KEYS,
        how="outer",
        suffixes=suffixes,
        indicator=True,
        validate="one_to_one"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    files = {
        "registration": RAW / "studentRegistration.csv",
        "student_info": RAW / "studentInfo.csv",
        "weekly_activity": INTERIM / "weekly_activity.parquet",
        "corrected": INTERIM / "corrected_week4_final_dataset.parquet",
        "combined": INTERIM / "week4_combined_features_v4.parquet",
        "panel": INTERIM / "weekly_panel_weeks0_4.parquet",
    }

    required = ["registration", "student_info", "weekly_activity",
                "corrected", "combined"]
    missing_files = [str(files[k]) for k in required if not files[k].exists()]
    if missing_files:
        raise FileNotFoundError(
            "No se puede ejecutar la auditoría. Faltan estos archivos:\n"
            + "\n".join(missing_files)
        )

    registration = pd.read_csv(files["registration"], na_values=["?"])
    student_info = pd.read_csv(files["student_info"])
    weekly = pd.read_parquet(files["weekly_activity"])
    corrected = pd.read_parquet(files["corrected"])
    combined = pd.read_parquet(files["combined"])

    require_columns(registration, KEYS + ["date_registration", "date_unregistration"],
                    "studentRegistration.csv")
    require_columns(student_info, KEYS + ["final_result"], "studentInfo.csv")
    require_columns(weekly, KEYS + ["week", "weekly_clicks", "active_days",
                                    "activity_records"], "weekly_activity.parquet")
    require_columns(corrected, KEYS + ["target_future_withdrawal"],
                    "corrected_week4_final_dataset.parquet")
    require_columns(combined, KEYS + ["target_future_withdrawal"],
                    "week4_combined_features_v4.parquet")

    checks = []
    integrity = []
    for name, frame, key_cols in [
        ("registration", registration, KEYS),
        ("student_info", student_info, KEYS),
        ("weekly_activity", weekly, KEYS + ["week"]),
        ("corrected", corrected, KEYS),
        ("combined", combined, KEYS),
    ]:
        integrity.append({
            "archivo": name,
            "filas": len(frame),
            "matriculas_unicas": frame[KEYS].drop_duplicates().shape[0],
            "duplicados_clave": int(frame.duplicated(key_cols).sum()),
            "columnas": len(frame.columns)
        })
        add_check(checks, f"duplicados_clave_{name}",
                  int(frame.duplicated(key_cols).sum()),
                  "OK" if not frame.duplicated(key_cols).any() else "ERROR")
    save(pd.DataFrame(integrity), "142_integridad_archivos.csv")

    # A. Integridad de las claves de matrícula entre las dos fuentes originales.
    reg_keys = registration[KEYS].drop_duplicates()
    info_keys = student_info[KEYS].drop_duplicates()
    key_audit = reg_keys.merge(info_keys, on=KEYS, how="outer",
                               indicator=True, validate="one_to_one")
    key_counts = key_audit["_merge"].value_counts().rename_axis("resultado").reset_index(name="matriculas")
    save(key_counts, "142_cruce_registration_studentinfo.csv")
    add_check(checks, "matriculas_sin_studentInfo",
              int((key_audit["_merge"] == "left_only").sum()),
              "OK" if not (key_audit["_merge"] == "left_only").any() else "REVISAR")
    add_check(checks, "matriculas_sin_registration",
              int((key_audit["_merge"] == "right_only").sum()),
              "OK" if not (key_audit["_merge"] == "right_only").any() else "REVISAR")

    # B. Fechas y reconstrucción exacta del objetivo del script 23 (corte día 34).
    registration["date_registration"] = pd.to_numeric(
        registration["date_registration"], errors="coerce")
    registration["date_unregistration"] = pd.to_numeric(
        registration["date_unregistration"], errors="coerce")

    source = registration.merge(
        student_info[KEYS + ["final_result"]],
        on=KEYS, how="left", validate="one_to_one", indicator=True
    )

    source["target_rebuilt"] = pd.NA
    future = source["date_unregistration"].notna() & (source["date_unregistration"] > 34)
    no_future = (
        source["final_result"].isin(["Pass", "Fail", "Distinction"])
        & source["date_unregistration"].isna()
    )
    source.loc[future, "target_rebuilt"] = 1
    source.loc[no_future, "target_rebuilt"] = 0

    valid_registration = (
        source["date_registration"].notna()
        & (source["date_registration"] <= 34)
    )
    early_withdrawal = (
        source["date_unregistration"].notna()
        & (source["date_unregistration"] <= 34)
    )
    eligible = source[
        valid_registration & ~early_withdrawal & source["target_rebuilt"].notna()
    ].copy()
    eligible["target_rebuilt"] = eligible["target_rebuilt"].astype(int)

    target_source = eligible[KEYS + ["target_rebuilt", "date_registration",
                                     "date_unregistration", "final_result"]].copy()
    target_corrected = corrected[KEYS + ["target_future_withdrawal"]].copy()
    target_combined = combined[KEYS + ["target_future_withdrawal"]].copy()

    for label, frame in [("corrected", target_corrected), ("combined", target_combined)]:
        comp = target_source[KEYS + ["target_rebuilt"]].merge(
            frame, on=KEYS, how="outer", indicator=True, validate="one_to_one"
        )
        comp["target_equal"] = (
            pd.to_numeric(comp["target_rebuilt"], errors="coerce")
            == pd.to_numeric(comp["target_future_withdrawal"], errors="coerce")
        )
        save(comp, f"142_comparacion_objetivo_{label}.csv")
        differences = comp[
            (comp["_merge"] != "both") | (~comp["target_equal"].fillna(False))
        ]
        save(differences, f"142_discrepancias_objetivo_{label}.csv")
        add_check(checks, f"discrepancias_objetivo_{label}", len(differences),
                  "OK" if len(differences) == 0 else "ERROR")

    save(
        pd.crosstab(eligible["final_result"], eligible["target_rebuilt"],
                    margins=True).reset_index(),
        "142_target_por_resultado_final.csv"
    )
    fail_future = eligible[
        (eligible["final_result"] == "Fail") &
        (eligible["target_rebuilt"] == 1)
    ].copy()
    save(fail_future, "142_casos_fail_con_retiro_futuro.csv")
    add_check(
        checks, "registros_final_result_Fail_con_retiro_posterior",
        len(fail_future),
        "REVISAR DEFINICIÓN" if len(fail_future) else "OK",
        "Estos casos reciben target=1 por fecha de baja > día 34 aunque final_result sea Fail."
    )

    date_summary = pd.DataFrame([
        {"control": "date_registration_missing", "registros": int(source["date_registration"].isna().sum())},
        {"control": "date_registration_before_day_0", "registros": int((source["date_registration"] < 0).sum())},
        {"control": "date_registration_after_day_0", "registros": int((source["date_registration"] > 0).sum())},
        {"control": "date_registration_after_day_34", "registros": int((source["date_registration"] > 34).sum())},
        {"control": "date_unregistration_missing", "registros": int(source["date_unregistration"].isna().sum())},
        {"control": "date_unregistration_equal_minus_1", "registros": int((source["date_unregistration"] == -1).sum())},
        {"control": "date_unregistration_le_day_34", "registros": int((source["date_unregistration"] <= 34).sum())},
        {"control": "date_unregistration_gt_day_34", "registros": int((source["date_unregistration"] > 34).sum())},
    ])
    save(date_summary, "142_auditoria_fechas.csv")

    registration_sensitivity = []
    for name, sample in [
        ("cohorte_actual_registration_le_34", eligible),
        ("matricula_desde_dia_0_registration_ge_0",
         eligible[eligible["date_registration"] >= 0]),
        ("excluir_matricula_posterior_dia_0_registration_le_0",
         eligible[eligible["date_registration"] <= 0]),
    ]:
        registration_sensitivity.append({
            "muestra": name,
            "matriculas": len(sample),
            "retiros_futuros": int(sample["target_rebuilt"].sum()),
            "sin_retiro_futuro": int((sample["target_rebuilt"] == 0).sum()),
            "proporcion_retiro_futuro": float(sample["target_rebuilt"].mean()) if len(sample) else np.nan
        })
    save(pd.DataFrame(registration_sensitivity), "142_sensibilidad_fecha_matricula.csv")

    # C. Validez de weekly_activity y sus magnitudes.
    weekly_checks = []
    week_numeric = pd.to_numeric(weekly["week"], errors="coerce")
    numeric_rules = [
        ("week_missing_or_not_numeric", week_numeric.isna()),
        ("week_not_integer", week_numeric.notna() & (week_numeric % 1 != 0)),
        ("weekly_clicks_negative", weekly["weekly_clicks"] < 0),
        ("active_days_less_than_1", weekly["active_days"] < 1),
        ("active_days_greater_than_7", weekly["active_days"] > 7),
        ("activity_records_less_than_1", weekly["activity_records"] < 1),
    ]
    for name, mask in numeric_rules:
        count = int(mask.fillna(False).sum())
        weekly_checks.append({"control": name, "registros": count})
        add_check(checks, name, count, "OK" if count == 0 else "ERROR")
    weekly_checks.extend([
        {"control": "weekly_clicks_missing", "registros": int(weekly["weekly_clicks"].isna().sum())},
        {"control": "active_days_missing", "registros": int(weekly["active_days"].isna().sum())},
        {"control": "activity_records_missing", "registros": int(weekly["activity_records"].isna().sum())},
        {"control": "semanas_min", "registros": int(weekly["week"].min())},
        {"control": "semanas_max", "registros": int(weekly["week"].max())},
    ])
    save(pd.DataFrame(weekly_checks), "142_validacion_weekly_activity.csv")

    # D. Reconstrucción independiente del panel semanas 0-4.
    # La presencia de una fila estudiante-semana determina disponibilidad;
    # un cero de clics observado NO es lo mismo que una semana sin fila.
    weekly_0_4 = weekly[weekly["week"].isin(WEEKS)].copy()
    weekly_0_4["week"] = weekly_0_4["week"].astype(int)
    wide_clicks = weekly_0_4.pivot(
        index=KEYS, columns="week", values="weekly_clicks"
    ).reindex(columns=WEEKS)
    wide_clicks.columns = CLICK_COLS
    wide_flags = weekly_0_4.assign(_available=1).pivot(
        index=KEYS, columns="week", values="_available"
    ).reindex(columns=WEEKS)
    wide_flags.columns = AVAIL_COLS
    weekly_panel = wide_clicks.join(wide_flags).reset_index()

    rebuilt_panel = target_combined.merge(
        weekly_panel, on=KEYS, how="left", validate="one_to_one"
    )
    for col in AVAIL_COLS:
        rebuilt_panel[col] = rebuilt_panel[col].fillna(0).astype("int8")
    rebuilt_panel["target_future_withdrawal"] = pd.to_numeric(
        rebuilt_panel["target_future_withdrawal"], errors="raise"
    ).astype(int)

    # Mantener NaN en clics para semanas sin registros; no imputar cero.
    rebuilt_panel.to_parquet(OUT / "142_panel_reconstruido_weeks0_4.parquet", index=False)

    coverage = []
    for week, clicks, flag in zip(WEEKS, CLICK_COLS, AVAIL_COLS):
        coverage.append({
            "semana": week,
            "matriculas_panel": len(rebuilt_panel),
            "semanas_con_fila": int(rebuilt_panel[flag].sum()),
            "semanas_sin_fila": int((rebuilt_panel[flag] == 0).sum()),
            "clics_no_faltantes": int(rebuilt_panel[clicks].notna().sum()),
            "clics_faltantes": int(rebuilt_panel[clicks].isna().sum()),
            "clics_observados_iguales_a_cero": int((rebuilt_panel[clicks] == 0).sum()),
            "discordancia_flag_vs_clics": int(
                ((rebuilt_panel[flag] == 1) != rebuilt_panel[clicks].notna()).sum()
            )
        })
    save(pd.DataFrame(coverage), "142_cobertura_panel_semanal.csv")

    # E. Comparar panel reconstruido con el utilizado en los análisis 130-141.
    if files["panel"].exists():
        old_panel = pd.read_parquet(files["panel"])
        require_columns(old_panel, KEYS + ["target_future_withdrawal"] +
                        CLICK_COLS + AVAIL_COLS, "weekly_panel_weeks0_4.parquet")
        if old_panel.duplicated(KEYS).any():
            raise ValueError("weekly_panel_weeks0_4.parquet tiene claves duplicadas.")

        panel_compare = compare_frames(
            rebuilt_panel, old_panel,
            ["target_future_withdrawal"] + PANEL_COLS
        )
        save(panel_compare, "142_comparacion_detallada_panel.csv")

        differences = []
        both = panel_compare[panel_compare["_merge"] == "both"].copy()
        for col in ["target_future_withdrawal"] + PANEL_COLS:
            a = both[f"{col}_reconstruido"]
            b = both[f"{col}_existente"]
            same = a.eq(b) | (a.isna() & b.isna())
            differences.append({
                "variable": col,
                "comparaciones": len(both),
                "coincidencias": int(same.sum()),
                "diferencias": int((~same).sum()),
                "faltantes_reconstruido": int(a.isna().sum()),
                "faltantes_existente": int(b.isna().sum())
            })
        save(pd.DataFrame(differences), "142_resumen_diferencias_panel.csv")

        key_diff = int((panel_compare["_merge"] != "both").sum())
        target_diff = int(
            differences[0]["diferencias"]
        )
        feature_diff = int(sum(x["diferencias"] for x in differences[1:]))
        add_check(checks, "claves_panel_no_coincidentes", key_diff,
                  "OK" if key_diff == 0 else "ERROR")
        add_check(checks, "diferencias_target_panel", target_diff,
                  "OK" if target_diff == 0 else "ERROR")
        add_check(checks, "diferencias_variables_panel", feature_diff,
                  "OK" if feature_diff == 0 else "ERROR")
    else:
        add_check(checks, "panel_existente_disponible", 0, "NO DISPONIBLE",
                  "No se encontró weekly_panel_weeks0_4.parquet; se reconstruyó desde sus fuentes.")

    # F. Evaluaciones programadas: comprobar si es constante en cada grupo.
    if "evaluaciones_programadas_34" in combined.columns:
        course_context = (
            combined.groupby(["code_module", "code_presentation"])
            .agg(
                matriculas=("target_future_withdrawal", "size"),
                valores_distintos=("evaluaciones_programadas_34", "nunique"),
                minimo=("evaluaciones_programadas_34", "min"),
                maximo=("evaluaciones_programadas_34", "max")
            ).reset_index()
        )
        course_context["constante_en_grupo"] = course_context["valores_distintos"] == 1
        save(course_context, "142_constancia_evaluaciones_programadas.csv")

    # G. Resumen reproducible y estado general.
    summary = [
        {"control": "matriculas_elegibles_reconstruidas", "valor": len(eligible)},
        {"control": "matriculas_dataset_corregido", "valor": len(corrected)},
        {"control": "matriculas_dataset_combinado", "valor": len(combined)},
        {"control": "target_0_reconstruido", "valor": int((eligible["target_rebuilt"] == 0).sum())},
        {"control": "target_1_reconstruido", "valor": int((eligible["target_rebuilt"] == 1).sum())},
        {"control": "filas_weekly_activity", "valor": len(weekly)},
        {"control": "filas_panel_reconstruido", "valor": len(rebuilt_panel)},
        {"control": "duplicados_panel_reconstruido", "valor": int(rebuilt_panel.duplicated(KEYS).sum())},
        {"control": "python_version", "valor": platform.python_version()},
        {"control": "pandas_version", "valor": pd.__version__},
        {"control": "numpy_version", "valor": np.__version__},
    ]
    save(pd.DataFrame(summary), "142_resumen_ejecutivo.csv")
    save(pd.DataFrame(checks), "142_estado_controles.csv")

    print("\nAUDITORÍA 142 FINALIZADA")
    print("Matrículas elegibles reconstruidas:", len(eligible))
    print("Dataset combinado:", len(combined))
    print("Panel reconstruido:", len(rebuilt_panel))
    print("Controles:", len(checks))
    print("Resultados guardados en:", OUT)
    print("IMPORTANTE: revisar 142_estado_controles.csv y los archivos de discrepancias.")


if __name__ == "__main__":
    main()
