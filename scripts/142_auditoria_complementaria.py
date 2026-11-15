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
import sklearn
from sklearn.model_selection import train_test_split

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
        "student_vle": RAW / "studentVle.csv",
        "registration": RAW / "studentRegistration.csv",
        "student_info": RAW / "studentInfo.csv",
        "weekly_activity": INTERIM / "weekly_activity.parquet",
        "corrected": INTERIM / "corrected_week4_final_dataset.parquet",
        "enriched": INTERIM / "week4_enriched_features.parquet",
        "assessment_features": INTERIM / "week4_assessment_features_v4.parquet",
        "combined": INTERIM / "week4_combined_features_v4.parquet",
        "panel": INTERIM / "weekly_panel_weeks0_4.parquet",
    }

    required = ["student_vle", "registration", "student_info", "weekly_activity",
                "corrected", "enriched", "assessment_features", "combined"]
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
    enriched = pd.read_parquet(files["enriched"])
    assessment_features = pd.read_parquet(files["assessment_features"])
    combined = pd.read_parquet(files["combined"])

    require_columns(registration, KEYS + ["date_registration", "date_unregistration"],
                    "studentRegistration.csv")
    require_columns(student_info, KEYS + ["final_result"], "studentInfo.csv")
    require_columns(weekly, KEYS + ["week", "weekly_clicks", "active_days",
                                    "activity_records"], "weekly_activity.parquet")
    require_columns(corrected, KEYS + ["target_future_withdrawal"],
                    "corrected_week4_final_dataset.parquet")
    require_columns(enriched, KEYS + ["target_future_withdrawal"],
                    "week4_enriched_features.parquet")
    assessment_columns = [
        "evaluaciones_programadas_34",
        "evaluaciones_entregadas_34",
        "promedio_notas_34",
        "evaluaciones_aprobadas_34",
        "proporcion_entregadas_34",
        "entregas_tempranas_totales"
    ]
    require_columns(assessment_features, KEYS + assessment_columns,
                    "week4_assessment_features_v4.parquet")
    require_columns(combined, KEYS + ["target_future_withdrawal"],
                    "week4_combined_features_v4.parquet")

    checks = []
    integrity = []
    for name, frame, key_cols in [
        ("registration", registration, KEYS),
        ("student_info", student_info, KEYS),
        ("weekly_activity", weekly, KEYS + ["week"]),
        ("corrected", corrected, KEYS),
        ("enriched", enriched, KEYS),
        ("assessment_features", assessment_features, KEYS),
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

    # Reconstruir las variables de actividad de los scripts 23 y 25
    # directamente desde weekly_activity y comprobarlas contra enriched.
    activity_0_4 = weekly[weekly["week"].between(0, 4)].copy()

    def pivot_activity(metric, prefix):
        table = activity_0_4.pivot_table(
            index=KEYS, columns="week", values=metric, aggfunc="sum"
        ).reindex(columns=WEEKS)
        table.columns = [f"{prefix}_{int(week)}" for week in WEEKS]
        return table

    clicks_wide = pivot_activity("weekly_clicks", "clics_semana")
    days_wide = pivot_activity("active_days", "dias_semana")
    records_wide = pivot_activity("activity_records", "registros_semana")
    activity_features = clicks_wide.join(days_wide).join(records_wide).reset_index()

    click_feature_cols = [f"clics_semana_{week}" for week in WEEKS]
    day_feature_cols = [f"dias_semana_{week}" for week in WEEKS]

    activity_features["semanas_con_actividad"] = (
        activity_features[click_feature_cols].notna().sum(axis=1)
    )
    activity_features["clics_acumulados_0_4"] = (
        activity_features[click_feature_cols].sum(axis=1, min_count=1)
    )
    activity_features["promedio_clics_0_4"] = (
        activity_features[click_feature_cols].mean(axis=1)
    )
    activity_features["dias_activos_0_4"] = (
        activity_features[day_feature_cols].sum(axis=1, min_count=1)
    )
    activity_features["clics_recientes_3_4"] = (
        activity_features[["clics_semana_3", "clics_semana_4"]]
        .sum(axis=1, min_count=1)
    )
    activity_features["clics_anteriores_0_2"] = (
        activity_features[["clics_semana_0", "clics_semana_1",
                           "clics_semana_2"]]
        .sum(axis=1, min_count=1)
    )

    expected_enriched = corrected.merge(
        activity_features, on=KEYS, how="left", validate="one_to_one"
    )
    require_columns(
        enriched, expected_enriched.columns.tolist(),
        "week4_enriched_features.parquet"
    )
    enriched_compare = expected_enriched.merge(
        enriched, on=KEYS, how="outer",
        suffixes=("_reconstruido", "_existente"),
        indicator=True, validate="one_to_one"
    )
    enriched_rows = []
    both_enriched = enriched_compare[enriched_compare["_merge"] == "both"]
    for col in expected_enriched.columns:
        if col in KEYS:
            continue
        left_values = both_enriched[f"{col}_reconstruido"]
        right_values = both_enriched[f"{col}_existente"]
        same = left_values.eq(right_values) | (
            left_values.isna() & right_values.isna()
        )
        enriched_rows.append({
            "variable": col,
            "comparaciones": len(both_enriched),
            "coincidencias": int(same.sum()),
            "diferencias": int((~same).sum())
        })
    enriched_key_diff = int((enriched_compare["_merge"] != "both").sum())
    save(pd.DataFrame(enriched_rows), "142_comparacion_features_enriquecidas.csv")
    save(enriched_compare[enriched_compare["_merge"] != "both"],
         "142_discrepancias_features_enriquecidas.csv")
    add_check(
        checks, "claves_features_enriquecidas_no_coincidentes",
        enriched_key_diff,
        "OK" if enriched_key_diff == 0 else "ERROR"
    )
    for item in enriched_rows:
        add_check(
            checks, f"diferencias_feature_enriquecida_{item['variable']}",
            item["diferencias"],
            "OK" if item["diferencias"] == 0 else "ERROR"
        )

    # Verificar específicamente las variables de semana 3 y 4 creadas en script 23.
    direct_checks = [
        ("week3_clicks", "clics_semana_3"),
        ("week4_clicks", "clics_semana_4"),
        ("week4_active_days", "dias_semana_4"),
        ("week4_activity_records", "registros_semana_4")
    ]
    direct_rows = []
    for model_col, weekly_col in direct_checks:
        left_values = expected_enriched[model_col]
        right_values = expected_enriched[weekly_col]
        same = left_values.eq(right_values) | (
            left_values.isna() & right_values.isna()
        )
        direct_rows.append({
            "variable": model_col,
            "fuente_semanal": weekly_col,
            "coincidencias": int(same.sum()),
            "diferencias": int((~same).sum())
        })
    expected_has_week4 = expected_enriched["registros_semana_4"].notna().astype(int)
    same_has = expected_enriched["has_week4_activity"].eq(expected_has_week4)
    direct_rows.append({
        "variable": "has_week4_activity",
        "fuente_semanal": "registros_semana_4.notna()",
        "coincidencias": int(same_has.sum()),
        "diferencias": int((~same_has).sum())
    })
    expected_change = (
        expected_enriched["week4_clicks"] - expected_enriched["week3_clicks"]
    )
    expected_change = expected_change.mask(
        expected_enriched["week3_clicks"].isna()
        | expected_enriched["week4_clicks"].isna()
    )
    same_change = expected_enriched["weekly_change"].eq(expected_change) | (
        expected_enriched["weekly_change"].isna() & expected_change.isna()
    )
    direct_rows.append({
        "variable": "weekly_change",
        "fuente_semanal": "week4_clicks - week3_clicks",
        "coincidencias": int(same_change.sum()),
        "diferencias": int((~same_change).sum())
    })
    save(pd.DataFrame(direct_rows), "142_comprobacion_features_predictivas.csv")
    for item in direct_rows:
        add_check(
            checks, f"diferencias_feature_predictiva_{item['variable']}",
            item["diferencias"],
            "OK" if item["diferencias"] == 0 else "ERROR"
        )

    # Reconstruir el merge del script 49 y comparar todas sus columnas.
    expected_combined = enriched.merge(
        assessment_features[KEYS + assessment_columns],
        on=KEYS, how="left", validate="one_to_one"
    )
    require_columns(
        combined, expected_combined.columns.tolist(),
        "week4_combined_features_v4.parquet"
    )
    pipeline_compare = expected_combined.merge(
        combined,
        on=KEYS, how="outer", suffixes=("_reconstruido", "_existente"),
        indicator=True, validate="one_to_one"
    )
    pipeline_rows = []
    both_pipeline = pipeline_compare[pipeline_compare["_merge"] == "both"]
    for col in expected_combined.columns:
        if col in KEYS:
            continue
        left_col = f"{col}_reconstruido"
        right_col = f"{col}_existente"
        if left_col not in pipeline_compare.columns or right_col not in pipeline_compare.columns:
            continue
        left_values = both_pipeline[left_col]
        right_values = both_pipeline[right_col]
        same = left_values.eq(right_values) | (left_values.isna() & right_values.isna())
        pipeline_rows.append({
            "variable": col,
            "comparaciones": len(both_pipeline),
            "coincidencias": int(same.sum()),
            "diferencias": int((~same).sum())
        })
    pipeline_key_diff = int((pipeline_compare["_merge"] != "both").sum())
    save(pd.DataFrame(pipeline_rows), "142_comparacion_pipeline_v4.csv")
    save(pipeline_compare[pipeline_compare["_merge"] != "both"],
         "142_discrepancias_pipeline_v4.csv")
    add_check(
        checks, "claves_pipeline_v4_no_coincidentes", pipeline_key_diff,
        "OK" if pipeline_key_diff == 0 else "ERROR"
    )
    for item in pipeline_rows:
        add_check(
            checks, f"diferencias_pipeline_v4_{item['variable']}",
            item["diferencias"],
            "OK" if item["diferencias"] == 0 else "ERROR"
        )

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

    # D. Reconstrucción de weekly_activity desde el studentVle original.
    # Se procesa por bloques para no cargar los 10.6 millones de registros
    # originales completos en memoria.
    raw_partials = []
    raw_columns = KEYS + ["date", "sum_click"]
    for chunk in pd.read_csv(
        files["student_vle"], usecols=raw_columns, chunksize=750000
    ):
        chunk["week"] = (chunk["date"] // 7).astype("int32")
        partial = (
            chunk.groupby(KEYS + ["week"], as_index=False)
            .agg(
                weekly_clicks=("sum_click", "sum"),
                activity_records=("sum_click", "size"),
                observed_dates=("date", lambda values: tuple(pd.unique(values)))
            )
        )
        raw_partials.append(partial)

    raw_grouped = pd.concat(raw_partials, ignore_index=True)
    raw_weekly = (
        raw_grouped.groupby(KEYS + ["week"], as_index=False)
        .agg(
            weekly_clicks=("weekly_clicks", "sum"),
            activity_records=("activity_records", "sum"),
            active_days=("observed_dates",
                         lambda values: len(set().union(*values)))
        )
    )

    raw_comparison = raw_weekly.merge(
        weekly[KEYS + ["week", "weekly_clicks", "active_days", "activity_records"]],
        on=KEYS + ["week"], how="outer",
        suffixes=("_raw", "_parquet"), indicator=True,
        validate="one_to_one"
    )
    raw_diff_rows = []
    raw_both = raw_comparison[raw_comparison["_merge"] == "both"]
    for col in ["weekly_clicks", "active_days", "activity_records"]:
        a = raw_both[f"{col}_raw"]
        b = raw_both[f"{col}_parquet"]
        same = a.eq(b) | (a.isna() & b.isna())
        raw_diff_rows.append({
            "variable": col,
            "coincidencias": int(same.sum()),
            "diferencias": int((~same).sum())
        })
    raw_key_diff = int((raw_comparison["_merge"] != "both").sum())
    save(raw_comparison[raw_comparison["_merge"] != "both"],
         "142_discrepancias_claves_raw_vs_weekly.csv")
    save(pd.DataFrame(raw_diff_rows), "142_comparacion_raw_vs_weekly.csv")
    add_check(
        checks, "claves_raw_vs_weekly_no_coincidentes", raw_key_diff,
        "OK" if raw_key_diff == 0 else "ERROR"
    )
    for item in raw_diff_rows:
        add_check(
            checks, f"diferencias_raw_vs_weekly_{item['variable']}",
            item["diferencias"],
            "OK" if item["diferencias"] == 0 else "ERROR"
        )

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

    # E. Reproducir los tamaños de muestra de los análisis longitudinales.
    # Esto permite detectar si los filtros publicados cambian al reconstruir
    # el panel desde los datos originales.
    analysis_panel = rebuilt_panel.merge(
        eligible[KEYS + ["date_registration"]],
        on=KEYS, how="left", validate="one_to_one"
    )
    analysis_panel["semanas_observadas"] = (
        analysis_panel[CLICK_COLS].notna().sum(axis=1)
    )
    analysis_panel["media_actividad"] = (
        analysis_panel[CLICK_COLS].mean(axis=1)
    )
    analysis_panel["sd_actividad"] = (
        analysis_panel[CLICK_COLS].std(axis=1, ddof=1)
    )
    analysis_panel["cv_actividad"] = (
        analysis_panel["sd_actividad"] / analysis_panel["media_actividad"]
    )

    cv_sample = (
        (analysis_panel["semanas_observadas"] >= 2)
        & (analysis_panel["media_actividad"] > 0)
        & analysis_panel["sd_actividad"].notna()
        & analysis_panel["cv_actividad"].notna()
    )
    complete_sample = analysis_panel[CLICK_COLS].notna().all(axis=1)

    sample_masks = {
        "cohorte_completa": pd.Series(True, index=analysis_panel.index),
        "muestra_CV_scripts_124_125_128_129": cv_sample,
        "trayectorias_completas_scripts_131_132": complete_sample,
        "matricula_posterior_dia_0": analysis_panel["date_registration"] > 0,
        "matricula_hasta_dia_0": analysis_panel["date_registration"] <= 0,
    }

    sample_rows = []
    for sample_name, mask in sample_masks.items():
        subset = analysis_panel[mask]
        for target_value in [0, 1]:
            group = subset[subset["target_future_withdrawal"] == target_value]
            sample_rows.append({
                "muestra": sample_name,
                "target": target_value,
                "matriculas": len(group),
                "proporcion_dentro_de_muestra": (
                    len(group) / len(subset) if len(subset) else np.nan
                ),
                "media_semanas_observadas": (
                    group["semanas_observadas"].mean() if len(group) else np.nan
                )
            })
    save(pd.DataFrame(sample_rows), "142_reconstruccion_muestras_analiticas.csv")

    add_check(
        checks, "n_muestra_CV_esperado_24887", int(cv_sample.sum()),
        "OK" if int(cv_sample.sum()) == 24887 else "REVISAR",
        "El valor esperado procede de las tablas del experimento 124."
    )
    add_check(
        checks, "n_trayectorias_completas_esperado_15317",
        int(complete_sample.sum()),
        "OK" if int(complete_sample.sum()) == 15317 else "REVISAR",
        "El valor esperado procede de los experimentos 132 y 138."
    )

    # F. Auditar solapamiento de personas entre periodos y dentro de 2013J.
    # El split original de 2013J se replica con el mismo random_state y
    # estratificación usados en los scripts 107-114.
    period_data = combined.copy()
    period_data["id_student"] = period_data["id_student"].astype(str)

    train_period = period_data[period_data["code_presentation"] == "2013B"]
    validation_period = period_data[period_data["code_presentation"] == "2013J"]
    test_b_period = period_data[period_data["code_presentation"] == "2014B"]
    test_j_period = period_data[period_data["code_presentation"] == "2014J"]

    calibration_period, threshold_period = train_test_split(
        validation_period,
        test_size=0.5,
        random_state=42,
        stratify=validation_period["target_future_withdrawal"]
    )

    id_sets = {
        "train_2013B": set(train_period["id_student"]),
        "calibracion_2013J": set(calibration_period["id_student"]),
        "seleccion_umbral_2013J": set(threshold_period["id_student"]),
        "test_2014B": set(test_b_period["id_student"]),
        "test_2014J": set(test_j_period["id_student"])
    }

    overlap_pairs = [
        ("calibracion_vs_seleccion_2013J", "calibracion_2013J", "seleccion_umbral_2013J"),
        ("test_2014B_vs_train_2013B", "test_2014B", "train_2013B"),
        ("test_2014B_vs_calibracion_2013J", "test_2014B", "calibracion_2013J"),
        ("test_2014B_vs_seleccion_umbral_2013J", "test_2014B", "seleccion_umbral_2013J"),
        ("test_2014J_vs_train_2013B", "test_2014J", "train_2013B"),
        ("test_2014J_vs_calibracion_2013J", "test_2014J", "calibracion_2013J"),
        ("test_2014J_vs_seleccion_umbral_2013J", "test_2014J", "seleccion_umbral_2013J"),
        ("test_2014B_vs_test_2014J", "test_2014B", "test_2014J")
    ]

    overlap_rows = []
    for name, left, right in overlap_pairs:
        shared = id_sets[left].intersection(id_sets[right])
        overlap_rows.append({
            "comparacion": name,
            "personas_compartidas": len(shared),
            "ids_compartidos": ";".join(sorted(shared))
        })
    save(pd.DataFrame(overlap_rows), "142_solapamiento_estudiantes_periodos.csv")

    # G. Comparar panel reconstruido con el utilizado en los análisis 130-141.
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

    # H. Evaluaciones programadas: comprobar si es constante en cada grupo.
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

    # I. Resumen reproducible y estado general.
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
        {"control": "scikit_learn_version", "valor": sklearn.__version__},
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
