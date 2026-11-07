
# ============================================================
# EXPERIMENTO 141
# ROBUSTEZ DE LA PERSISTENCIA SEGUN ABANDONO FUTURO
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from statsmodels.stats.multitest import multipletests

# ------------------------------------------------------------
# 1. RUTAS
# ------------------------------------------------------------

BASE = Path(__file__).resolve().parent.parent

ARCHIVO_PANEL = (
    BASE / "data" / "interim" /
    "weekly_panel_weeks0_4.parquet"
)

ARCHIVO_TARGET = (
    BASE / "data" / "interim" /
    "week4_combined_features_v4.parquet"
)

CARPETA_SALIDA = BASE / "outputs" / "tables"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. CARGAR PANEL Y TARGET
# ------------------------------------------------------------

print("=" * 65)
print("EXPERIMENTO 141: ROBUSTEZ DE PERSISTENCIA")
print("=" * 65)

df = pd.read_parquet(ARCHIVO_PANEL)
df_target = pd.read_parquet(ARCHIVO_TARGET)

claves = [
    "code_module",
    "code_presentation",
    "id_student"
]

df_target = df_target[
    claves + ["target_future_withdrawal"]
].copy()

df_target = df_target.rename(
    columns={
        "target_future_withdrawal": "target"
    }
)

df = df.merge(
    df_target,
    on=claves,
    how="inner",
    validate="one_to_one"
)

df["grupo"] = (
    df["code_module"].astype(str)
    + "_"
    + df["code_presentation"].astype(str)
)

df["id_curso"] = (
    df["code_module"].astype(str)
    + "_"
    + df["code_presentation"].astype(str)
    + "_"
    + df["id_student"].astype(str)
)

print("\nDIMENSIONES DEL PANEL:", df.shape)
print("\nDISTRIBUCION DEL TARGET:")
print(df["target"].value_counts())

# ------------------------------------------------------------
# 3. CONSTRUIR PANEL LONGITUDINAL
# ------------------------------------------------------------

registros = []

for semana in range(5):

    columna = f"clics_semana_{semana}"

    temporal = df[
        claves + ["grupo", "id_curso", "target", columna]
    ].copy()

    temporal = temporal.rename(
        columns={columna: "clics"}
    )

    temporal["semana"] = semana

    temporal = temporal.dropna(
        subset=["clics", "target"]
    )

    registros.append(temporal)

long = pd.concat(
    registros,
    ignore_index=True
)

long["log_clics"] = np.log1p(long["clics"])

# Estandarizacion dentro de cada modulo-presentacion
long["media_grupo"] = (
    long.groupby("grupo")["log_clics"]
    .transform("mean")
)

long["sd_grupo"] = (
    long.groupby("grupo")["log_clics"]
    .transform("std")
)

long["actividad_relativa"] = (
    long["log_clics"] - long["media_grupo"]
) / long["sd_grupo"]

print("\nPANEL LONGITUDINAL:", long.shape)

# ------------------------------------------------------------
# 4. CONSTRUIR TRANSICIONES CONSECUTIVAS
# ------------------------------------------------------------

transiciones = []

for semana in range(4):

    actual = long[
        long["semana"] == semana
    ].copy()

    siguiente = long[
        long["semana"] == semana + 1
    ].copy()

    actual = actual[
        claves + [
            "id_curso",
            "grupo",
            "target",
            "actividad_relativa"
        ]
    ]

    siguiente = siguiente[
        claves + ["actividad_relativa"]
    ]

    par = actual.merge(
        siguiente,
        on=claves,
        how="inner",
        validate="one_to_one",
        suffixes=("_actual", "_siguiente")
    )

    par = par.rename(
        columns={
            "actividad_relativa_actual": "actividad_actual",
            "actividad_relativa_siguiente": "actividad_siguiente"
        }
    )

    par["transicion"] = semana

    transiciones.append(par)

datos = pd.concat(
    transiciones,
    ignore_index=True
)

print("\nOBSERVACIONES DE TRANSICIONES:", len(datos))
print(
    "ESTUDIANTES- CURSO:",
    datos["id_curso"].nunique()
)

# Numero de transiciones observadas por estudiante-curso
conteo = (
    datos.groupby("id_curso")["transicion"]
    .nunique()
    .rename("numero_transiciones")
)

datos = datos.merge(
    conteo,
    on="id_curso",
    how="left",
    validate="many_to_one"
)

# ------------------------------------------------------------
# 5. FUNCION PARA ESTIMAR CADA MODELO
# ------------------------------------------------------------

def ejecutar_modelo(datos_modelo, nombre_muestra):

    datos_modelo = datos_modelo.copy()

    # Indicadores de transicion
    for j in range(1, 4):
        datos_modelo[f"w{j}"] = (
            datos_modelo["transicion"] == j
        ).astype(int)

    # Interacciones con target
    for j in range(1, 4):
        datos_modelo[f"target_w{j}"] = (
            datos_modelo["target"]
            * datos_modelo[f"w{j}"]
        )

    # Interacciones con actividad
    for j in range(1, 4):
        datos_modelo[f"act_w{j}"] = (
            datos_modelo["actividad_actual"]
            * datos_modelo[f"w{j}"]
        )

    datos_modelo["act_target"] = (
        datos_modelo["actividad_actual"]
        * datos_modelo["target"]
    )

    for j in range(1, 4):
        datos_modelo[f"act_w{j}_target"] = (
            datos_modelo["actividad_actual"]
            * datos_modelo[f"w{j}"]
            * datos_modelo["target"]
        )

    formula = (
        "actividad_siguiente ~ "
        "actividad_actual + target + "
        "w1 + w2 + w3 + "
        "target_w1 + target_w2 + target_w3 + "
        "act_target + "
        "act_w1 + act_w2 + act_w3 + "
        "act_w1_target + act_w2_target + act_w3_target + "
        "C(grupo)"
    )

    modelo = smf.ols(
        formula=formula,
        data=datos_modelo
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": datos_modelo["id_student"]
        }
    )

    print("\n" + "=" * 65)
    print("MODELO:", nombre_muestra)
    print("=" * 65)

    print("Observaciones:", len(datos_modelo))
    print(
        "Estudiantes-curso:",
        datos_modelo["id_curso"].nunique()
    )
    print(
        "Estudiantes:",
        datos_modelo["id_student"].nunique()
    )
    print("R-cuadrado:", round(modelo.rsquared, 5))

    resultados = []


    # Pendientes por grupo y transicion
    for j in range(4):

        if j == 0:
            efecto_no = "actividad_actual"
            efecto_dif = "act_target"
        else:
            efecto_no = (
                f"actividad_actual + act_w{j}"
            )
            efecto_dif = (
                f"act_target + act_w{j}_target"
            )

        prueba_no = modelo.t_test(efecto_no)
        prueba_dif = modelo.t_test(efecto_dif)

        efecto_abandono = (
            efecto_no + " + " + efecto_dif
        )

        prueba_abandono = modelo.t_test(
            efecto_abandono
        )

        ci_no = prueba_no.conf_int()[0]
        ci_abandono = prueba_abandono.conf_int()[0]
        ci_dif = prueba_dif.conf_int()[0]

        # Guardar resultados dentro del ciclo
        resultados.append({
            "muestra": nombre_muestra,
            "transicion": f"{j} a {j + 1}",

            "rho_no_abandono": np.asarray(
                prueba_no.effect
            ).item(),

            "rho_abandono": np.asarray(
                prueba_abandono.effect
            ).item(),

            "diferencia": np.asarray(
                prueba_dif.effect
            ).item(),

            "error_estandar_diferencia": np.asarray(
                prueba_dif.sd
            ).item(),

            "ic95_diferencia_inferior": float(ci_dif[0]),
            "ic95_diferencia_superior": float(ci_dif[1]),

            "p_valor": np.asarray(
                prueba_dif.pvalue
            ).item(),

            "ic95_no_abandono_inferior": float(ci_no[0]),
            "ic95_no_abandono_superior": float(ci_no[1]),

            "ic95_abandono_inferior": float(ci_abandono[0]),
            "ic95_abandono_superior": float(ci_abandono[1])
        })

    # Prueba A: las cuatro diferencias son conjuntamente iguales a cero

    nombres = list(modelo.params.index)

    R = np.zeros((4, len(nombres)))

    for j in range(4):

        nombre_base = "act_target"

        if nombre_base in nombres:
            R[j, nombres.index(nombre_base)] = 1

        if j > 0:
            nombre_interaccion = f"act_w{j}_target"

            R[j, nombres.index(nombre_interaccion)] = 1

    prueba_global = modelo.wald_test(
        R,
        scalar=True
    )

    print("\nPRUEBA A: DIFERENCIAS CONJUNTAMENTE IGUALES A CERO")
    print("Chi-cuadrado:", float(prueba_global.statistic))
    print("Grados de libertad:", 4)
    print("p-valor:", float(prueba_global.pvalue))


    # Prueba B: igualdad de las diferencias entre transiciones

    R_igualdad = np.zeros((3, len(nombres)))

    R_igualdad[0, nombres.index("act_w1_target")] = 1
    R_igualdad[1, nombres.index("act_w2_target")] = 1
    R_igualdad[2, nombres.index("act_w3_target")] = 1

    prueba_igualdad = modelo.wald_test(
        R_igualdad,
        scalar=True
    )

    print("\nPRUEBA B: IGUALDAD DE LAS DIFERENCIAS ENTRE TRANSICIONES")
    print("Chi-cuadrado:", float(prueba_igualdad.statistic))
    print("Grados de libertad:", 3)
    print("p-valor:", float(prueba_igualdad.pvalue))

    tabla = pd.DataFrame(resultados)
    print(
        tabla[
            [
                "transicion",
                "rho_no_abandono",
                "rho_abandono",
                "diferencia",
                "p_valor"
            ]
        ].to_string(index=False)
    )

    return (
        resultados,
        float(prueba_global.statistic),
        float(prueba_global.pvalue),
        float(prueba_igualdad.statistic),
        float(prueba_igualdad.pvalue)
    )


# ------------------------------------------------------------
# 6. DEFINIR MUESTRAS DE ROBUSTEZ
# ------------------------------------------------------------

muestras = {
    "Al menos 2 transiciones": datos[
        datos["numero_transiciones"] >= 2
    ].copy(),

    "Al menos 3 transiciones": datos[
        datos["numero_transiciones"] >= 3
    ].copy(),

    "4 transiciones completas": datos[
        datos["numero_transiciones"] == 4
    ].copy()
}

todos_resultados = []
resultados_globales = []

for nombre, muestra in muestras.items():

    (
        resultados,
        chi2,
        p_global,
        chi2_igualdad,
        p_igualdad
    ) = ejecutar_modelo(
        muestra,
        nombre
    )

    todos_resultados.extend(resultados)

    resultados_globales.append({
        "muestra": nombre,
        "observaciones": len(muestra),
        "estudiantes_curso": muestra["id_curso"].nunique(),
        "estudiantes": muestra["id_student"].nunique(),

        # Prueba A
        "chi2_global": chi2,
        "grados_libertad": 4,
        "p_global": p_global,

        # Prueba B
        "chi2_igualdad": chi2_igualdad,
        "grados_libertad_igualdad": 3,
        "p_igualdad": p_igualdad
    })

# ------------------------------------------------------------
# 7. CORRECCION DE HOLM
# ------------------------------------------------------------

tabla_resultados = pd.DataFrame(todos_resultados)

tabla_resultados["p_holm"] = np.nan

for nombre in tabla_resultados["muestra"].unique():

    indices = tabla_resultados[
        tabla_resultados["muestra"] == nombre
    ].index

    pvalores = tabla_resultados.loc[
        indices,
        "p_valor"
    ].values

    _, p_ajustados, _, _ = multipletests(
        pvalores,
        method="holm"
    )

    tabla_resultados.loc[
        indices,
        "p_holm"
    ] = p_ajustados

tabla_global = pd.DataFrame(
    resultados_globales
)

# ------------------------------------------------------------
# 8. GUARDAR RESULTADOS
# ------------------------------------------------------------

archivo1 = (
    CARPETA_SALIDA /
    "141_robustez_persistencia_por_transicion.csv"
)

archivo2 = (
    CARPETA_SALIDA /
    "141_robustez_pruebas_globales.csv"
)

tabla_resultados.to_csv(
    archivo1,
    index=False,
    encoding="utf-8-sig"
)

tabla_global.to_csv(
    archivo2,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("RESUMEN DE PRUEBAS GLOBALES")
print("=" * 65)

print(tabla_global.to_string(index=False))

print("\nRESULTADOS INDIVIDUALES CON CORRECCION DE HOLM")
print(
    tabla_resultados[
        [
            "muestra",
            "transicion",
            "diferencia",
            "p_valor",
            "p_holm"
        ]
    ].to_string(index=False)
)

print("\nARCHIVOS GUARDADOS:")
print(archivo1)
print(archivo2)

print("\nEXPERIMENTO 141 FINALIZADO")
