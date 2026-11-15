"""
Auditoria complementaria del conjunto OULAD (experimento 142).

Este script NO modifica datasets ni resultados anteriores.
Reconstruye el panel semanal 0-4 y guarda auditorías independientes
en outputs/audit_142/.

Ejecutar desde la raíz del repositorio:
    python scripts/142_auditoria_complementaria.py
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "csv"
INTERIM = BASE / "data" / "interim"
OUT = BASE / "outputs" / "audit_142"

CLAVES = ["code_module", "code_presentation", "id_student"]
SEMANAS = list(range(5))
COLUMNAS_SEMANA = [f"clics_semana_{i}" for i in SEMANAS]


def guardar(df, nombre):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / nombre, index=False, encoding="utf-8-sig")


def exigir_columnas(df, columnas, nombre):
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"{nombre}: faltan columnas {faltan}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    archivos = {
        "registro": RAW / "studentRegistration.csv",
        "info": RAW / "studentInfo.csv",
        "actividad": INTERIM / "weekly_activity.parquet",
        "combinado": INTERIM / "week4_combined_features_v4.parquet",
        "panel_existente": INTERIM / "weekly_panel_weeks0_4.parquet",
    }

    for nombre in ["registro", "info", "actividad", "combinado"]:
        if not archivos[nombre].exists():
            raise FileNotFoundError(
                f"No se encuentra {archivos[nombre]}. "
                "Comprueba que los datos OULAD y los pasos previos estén disponibles."
            )

    registro = pd.read_csv(archivos["registro"], na_values=["?"])
    info = pd.read_csv(archivos["info"])
    actividad = pd.read_parquet(archivos["actividad"])
    combinado = pd.read_parquet(archivos["combinado"])

    exigir_columnas(registro, CLAVES + ["date_registration", "date_unregistration"],
                    "studentRegistration.csv")
    exigir_columnas(info, CLAVES + ["final_result"], "studentInfo.csv")
    exigir_columnas(actividad, CLAVES + ["week", "weekly_clicks"],
                    "weekly_activity.parquet")
    exigir_columnas(combinado, CLAVES + ["target_future_withdrawal"],
                    "week4_combined_features_v4.parquet")

    # ---------------------------------------------------------
    # 1. Integridad de claves y cardinalidad
    # ---------------------------------------------------------
    controles = []
    for nombre, datos in [
        ("registro", registro),
        ("info", info),
        ("actividad", actividad),
        ("combinado", combinado),
    ]:
        controles.append({
            "archivo": nombre,
            "filas": len(datos),
            "duplicados_clave": int(datos.duplicated(
                subset=CLAVES + (["week"] if nombre == "actividad" else [])
            ).sum()),
            "matriculas_unicas": int(datos[CLAVES].drop_duplicates().shape[0]),
        })

    guardar(pd.DataFrame(controles), "142_integridad_archivos.csv")

    if registro.duplicated(subset=CLAVES).any():
        raise ValueError("studentRegistration contiene claves de matrícula duplicadas.")
    if info.duplicated(subset=CLAVES).any():
        raise ValueError("studentInfo contiene claves de matrícula duplicadas.")
    if actividad.duplicated(subset=CLAVES + ["week"]).any():
        raise ValueError("weekly_activity contiene claves estudiante-semana duplicadas.")
    if combinado.duplicated(subset=CLAVES).any():
        raise ValueError("week4_combined_features_v4 contiene matrículas duplicadas.")

    # ---------------------------------------------------------
    # 2. Reconstruir el objetivo exactamente como el script 23
    # ---------------------------------------------------------
    registro["date_registration"] = pd.to_numeric(
        registro["date_registration"], errors="coerce"
    )
    registro["date_unregistration"] = pd.to_numeric(
        registro["date_unregistration"], errors="coerce"
    )

    fuente = registro.merge(
        info[CLAVES + ["final_result"]],
        on=CLAVES,
        how="left",
        validate="one_to_one",
        indicator=True
    )

    fuente["target_reconstruido"] = pd.NA

    retirada_futura = (
        fuente["date_unregistration"].notna()
        & (fuente["date_unregistration"] > 34)
    )
    sin_retirada = (
        fuente["final_result"].isin(["Pass", "Fail", "Distinction"])
        & fuente["date_unregistration"].isna()
    )

    fuente.loc[retirada_futura, "target_reconstruido"] = 1
    fuente.loc[sin_retirada, "target_reconstruido"] = 0

    matricula_valida = (
        fuente["date_registration"].notna()
        & (fuente["date_registration"] <= 34)
    )
    retirada_temprana = (
        fuente["date_unregistration"].notna()
        & (fuente["date_unregistration"] <= 34)
    )

    elegible = fuente[
        matricula_valida
        & ~retirada_temprana
        & fuente["target_reconstruido"].notna()
    ].copy()

    elegible["target_reconstruido"] = (
        elegible["target_reconstruido"].astype(int)
    )

    target_fuente = elegible[
        CLAVES + ["target_reconstruido", "date_registration",
                  "date_unregistration", "final_result"]
    ].copy()

    target_combinado = combinado[
        CLAVES + ["target_future_withdrawal"]
    ].copy()

    comparacion = target_fuente.merge(
        target_combinado,
        on=CLAVES,
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    comparacion["coincide_target"] = (
        comparacion["target_reconstruido"]
        == comparacion["target_future_withdrawal"]
    )

    guardar(
        comparacion.groupby(
            ["_merge", "coincide_target"], dropna=False
        ).size().reset_index(name="registros"),
        "142_comparacion_objetivo.csv"
    )

    discrepancias = comparacion[
        (comparacion["_merge"] != "both")
        | (~comparacion["coincide_target"].fillna(False))
    ].copy()
    guardar(discrepancias, "142_discrepancias_objetivo.csv")

    # Distribución cruzada de etiqueta y resultado final
    crosstab = pd.crosstab(
        elegible["final_result"],
        elegible["target_reconstruido"],
        margins=True
    ).reset_index()
    guardar(crosstab, "142_target_por_resultado_final.csv")

    # Casos de Fail que tienen retirada futura, para revisión de definición.
    casos_fail = elegible[
        (elegible["final_result"] == "Fail")
        & (elegible["target_reconstruido"] == 1)
    ].copy()
    guardar(casos_fail, "142_casos_fail_con_retiro_futuro.csv")

    # ---------------------------------------------------------
    # 3. Sensibilidad descriptiva a la fecha de matrícula
    # ---------------------------------------------------------
    resumen_matricula = []
    muestras = {
        "cohorte_actual_date_registration_le_34": elegible,
        "solo_matricula_desde_dia_0": elegible[
            elegible["date_registration"] >= 0
        ],
        "excluir_matricula_tardia_despues_dia_0": elegible[
            elegible["date_registration"] <= 0
        ],
    }

    for nombre, muestra in muestras.items():
        resumen_matricula.append({
            "muestra": nombre,
            "registros": len(muestra),
            "retiros_futuros": int(muestra["target_reconstruido"].sum()),
            "no_retiros_futuros": int((muestra["target_reconstruido"] == 0).sum()),
            "tasa_retiro_futuro": float(muestra["target_reconstruido"].mean())
                if len(muestra) else None,
            "matriculas_unicas": int(muestra[CLAVES].drop_duplicates().shape[0]),
        })

    guardar(pd.DataFrame(resumen_matricula), "142_sensibilidad_fecha_matricula.csv")

    fechas = pd.DataFrame([
        {
            "control": "date_registration_missing",
            "registros": int(fuente["date_registration"].isna().sum())
        },
        {
            "control": "date_registration_after_day_0",
            "registros": int((fuente["date_registration"] > 0).sum())
        },
        {
            "control": "date_registration_after_day_34",
            "registros": int((fuente["date_registration"] > 34).sum())
        },
        {
            "control": "date_unregistration_equal_minus_1",
            "registros": int((fuente["date_unregistration"] == -1).sum())
        },
        {
            "control": "date_unregistration_between_minus_34_and_34",
            "registros": int(
                fuente["date_unregistration"].between(-34, 34).sum()
            )
        },
    ])
    guardar(fechas, "142_auditoria_fechas.csv")

    # ---------------------------------------------------------
    # 4. Diagnóstico de evaluaciones programadas por módulo-presentación
    # ---------------------------------------------------------
    if "evaluaciones_programadas_34" in combinado.columns:
        contexto = (
            combinado.groupby(["code_module", "code_presentation"])
            .agg(
                registros=("target_future_withdrawal", "size"),
                valores_distintos=("evaluaciones_programadas_34", "nunique"),
                minimo=("evaluaciones_programadas_34", "min"),
                maximo=("evaluaciones_programadas_34", "max"),
            )
            .reset_index()
        )
        contexto["constante_en_grupo"] = contexto["valores_distintos"] == 1
        guardar(contexto, "142_constancia_evaluaciones_programadas.csv")

    # ---------------------------------------------------------
    # 5. Reconstruir panel 0-4 desde weekly_activity
    # ---------------------------------------------------------
    actividad_0_4 = actividad[
        actividad["week"].isin(SEMANAS)
    ].copy()

    panel_clicks = actividad_0_4.pivot(
        index=CLAVES,
        columns="week",
        values="weekly_clicks"
    )

    panel_clicks = panel_clicks.reindex(columns=SEMANAS)
    panel_clicks.columns = COLUMNAS_SEMANA
    panel_clicks = panel_clicks.reset_index()

    panel = target_combinado.merge(
        panel_clicks,
        on=CLAVES,
        how="left",
        validate="one_to_one"
    )

    panel["target_future_withdrawal"] = (
        panel["target_future_withdrawal"].astype(int)
    )

    panel.to_parquet(
        OUT / "142_panel_reconstruido_weeks0_4.parquet",
        index=False
    )

    resumen_panel = []
    for columna in COLUMNAS_SEMANA:
        resumen_panel.append({
            "semana": columna,
            "observaciones": int(panel[columna].notna().sum()),
            "faltantes": int(panel[columna].isna().sum()),
            "porcentaje_faltante": float(panel[columna].isna().mean() * 100),
            "ceros_observados": int((panel[columna] == 0).sum()),
        })
    guardar(pd.DataFrame(resumen_panel), "142_cobertura_panel_semanal.csv")

    # Comparar con el panel anterior si está disponible
    if archivos["panel_existente"].exists():
        panel_anterior = pd.read_parquet(archivos["panel_existente"])
        exigir_columnas(
            panel_anterior, CLAVES + COLUMNAS_SEMANA,
            "weekly_panel_weeks0_4.parquet"
        )

        if panel_anterior.duplicated(subset=CLAVES).any():
            raise ValueError("El panel existente contiene claves duplicadas.")

        cmp = panel[
            CLAVES + COLUMNAS_SEMANA
        ].merge(
            panel_anterior[CLAVES + COLUMNAS_SEMANA],
            on=CLAVES,
            how="outer",
            suffixes=("_nuevo", "_anterior"),
            indicator=True,
            validate="one_to_one"
        )

        filas = []
        filas.append({
            "control": "filas_panel_reconstruido",
            "valor": len(panel)
        })
        filas.append({
            "control": "filas_panel_existente",
            "valor": len(panel_anterior)
        })
        filas.append({
            "control": "claves_no_coincidentes",
            "valor": int((cmp["_merge"] != "both").sum())
        })

        for columna in COLUMNAS_SEMANA:
            nuevo = cmp[f"{columna}_nuevo"]
            anterior = cmp[f"{columna}_anterior"]
            iguales = (
                (nuevo.eq(anterior))
                | (nuevo.isna() & anterior.isna())
            )
            filas.append({
                "control": f"valores_iguales_{columna}",
                "valor": int(iguales.sum())
            })
            filas.append({
                "control": f"diferencias_{columna}",
                "valor": int((~iguales).sum())
            })

        guardar(pd.DataFrame(filas), "142_comparacion_panel_existente.csv")

    # ---------------------------------------------------------
    # 6. Resumen ejecutivo de la auditoría
    # ---------------------------------------------------------
    resumen = [
        ("matriculas_elegibles_reconstruidas", len(elegible)),
        ("matriculas_en_dataset_combinado", len(combinado)),
        ("discrepancias_de_claves_o_target", len(discrepancias)),
        ("target_0", int((elegible["target_reconstruido"] == 0).sum())),
        ("target_1", int((elegible["target_reconstruido"] == 1).sum())),
        ("panel_reconstruido_filas", len(panel)),
        ("panel_reconstruido_duplicados", int(panel.duplicated(subset=CLAVES).sum())),
    ]
    guardar(
        pd.DataFrame(resumen, columns=["control", "valor"]),
        "142_resumen_ejecutivo.csv"
    )

    print("\nAUDITORÍA 142 FINALIZADA")
    print("Matrículas elegibles reconstruidas:", len(elegible))
    print("Matrículas en dataset combinado:", len(combinado))
    print("Discrepancias de claves/target:", len(discrepancias))
    print("Panel reconstruido:", len(panel), "filas")
    print("Resultados guardados en:", OUT)


if __name__ == "__main__":
    main()
