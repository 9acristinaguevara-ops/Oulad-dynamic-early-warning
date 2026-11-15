import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# EXPERIMENTO 154:
# AUDITORIA TEMPORAL DE VARIABLES DE EVALUACION
#
# Reconstruye desde los CSV originales los dos predictores
# academicos usados por el modelo 150:
# - evaluaciones_programadas_34
# - evaluaciones_entregadas_34
#
# Comprueba que solo incluyan informacion disponible hasta
# el dia 34 y compara la reconstruccion con el parquet V4.
# No modifica datos ni resultados anteriores.
# =====================================================

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw" / "csv"
INTERIM = BASE / "data" / "interim"
SALIDA = BASE / "outputs" / "tables"
SALIDA.mkdir(parents=True, exist_ok=True)

CORTE = 34
CLAVES = ["code_module", "code_presentation", "id_student"]
CLAVES_CURSO = ["code_module", "code_presentation"]

ruta_assessments = RAW / "assessments.csv"
ruta_entregas = RAW / "studentAssessment.csv"
ruta_features = INTERIM / "week4_combined_features_v4.parquet"

for ruta in [ruta_assessments, ruta_entregas, ruta_features]:
    if not ruta.exists():
        raise FileNotFoundError("No se encontro el archivo: " + str(ruta))

assessments = pd.read_csv(ruta_assessments)
entregas = pd.read_csv(ruta_entregas)
features = pd.read_parquet(ruta_features)

assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")
entregas["date_submitted"] = pd.to_numeric(
    entregas["date_submitted"], errors="coerce"
)
entregas["score"] = pd.to_numeric(entregas["score"], errors="coerce")

# 1. Reconstruir las evaluaciones cuya fecha programada es hasta el dia 34.
programadas = assessments[
    assessments["date"].notna()
    & (assessments["date"] <= CORTE)
].copy()

conteo_programadas = programadas.groupby(CLAVES_CURSO).agg(
    programadas_reconstruidas=("id_assessment", "nunique")
).reset_index()

# 2. Unir entregas con fecha programada y filtrar estrictamente por ambos tiempos.
datos_entregas = entregas.merge(
    assessments[["id_assessment", "code_module", "code_presentation", "date"]],
    on="id_assessment",
    how="left",
    validate="many_to_one"
)

# Entregas tempranas de evaluaciones cuya fecha limite es posterior al corte.
tempranas_futuras = datos_entregas[
    datos_entregas["date_submitted"].notna()
    & (datos_entregas["date_submitted"] <= CORTE)
    & datos_entregas["date"].notna()
    & (datos_entregas["date"] > CORTE)
].copy()

# Registros con entrega <=34 y fecha programada <=34.
entregas_validas = datos_entregas[
    datos_entregas["date_submitted"].notna()
    & (datos_entregas["date_submitted"] <= CORTE)
    & datos_entregas["date"].notna()
    & (datos_entregas["date"] <= CORTE)
    & (datos_entregas["is_banked"] == 0)
].copy()

conteo_entregadas = entregas_validas.groupby(CLAVES).agg(
    entregadas_reconstruidas=("id_assessment", "nunique")
).reset_index()

# 3. Comparar reconstruccion con el dataset que uso el experimento 150.
necesarias = CLAVES + [
    "evaluaciones_programadas_34",
    "evaluaciones_entregadas_34"
]
faltantes = [c for c in necesarias if c not in features.columns]
if faltantes:
    raise ValueError("Faltan columnas en el parquet: " + ", ".join(faltantes))

base = features[necesarias].copy()
if base.duplicated(CLAVES).any():
    raise AssertionError("El parquet contiene matriculas duplicadas.")

base = base.merge(
    conteo_programadas,
    on=CLAVES_CURSO,
    how="left",
    validate="many_to_one"
)
base = base.merge(
    conteo_entregadas,
    on=CLAVES,
    how="left",
    validate="one_to_one"
)

base["programadas_reconstruidas"] = (
    base["programadas_reconstruidas"].fillna(0)
)
base["entregadas_reconstruidas"] = (
    base["entregadas_reconstruidas"].fillna(0)
)
base["evaluaciones_programadas_34"] = (
    pd.to_numeric(base["evaluaciones_programadas_34"], errors="coerce").fillna(0)
)
base["evaluaciones_entregadas_34"] = (
    pd.to_numeric(base["evaluaciones_entregadas_34"], errors="coerce").fillna(0)
)

base["diferencia_programadas"] = (
    base["evaluaciones_programadas_34"] - base["programadas_reconstruidas"]
)
base["diferencia_entregadas"] = (
    base["evaluaciones_entregadas_34"] - base["entregadas_reconstruidas"]
)

resumen = pd.DataFrame([
    {
        "variable": "evaluaciones_programadas_34",
        "matriculas_comparadas": len(base),
        "coincidencias": int((base["diferencia_programadas"] == 0).sum()),
        "diferencias": int((base["diferencia_programadas"] != 0).sum()),
        "diferencia_absoluta_maxima": float(
            base["diferencia_programadas"].abs().max()
        )
    },
    {
        "variable": "evaluaciones_entregadas_34",
        "matriculas_comparadas": len(base),
        "coincidencias": int((base["diferencia_entregadas"] == 0).sum()),
        "diferencias": int((base["diferencia_entregadas"] != 0).sum()),
        "diferencia_absoluta_maxima": float(
            base["diferencia_entregadas"].abs().max()
        )
    }
])

# 4. Comprobar que ningun predictor reconstruido use entregas posteriores al corte.
# Esta comprobacion es por definicion del filtro aplicado arriba.
registros_fuera = datos_entregas[
    datos_entregas["date_submitted"].notna()
    & (datos_entregas["date_submitted"] <= CORTE)
    & (
        datos_entregas["date"].isna()
        | (datos_entregas["date"] > CORTE)
    )
]
registros_fuera_no_banked = registros_fuera[
    registros_fuera["is_banked"] == 0
]

# 5. Resumen por presentacion para detectar errores concentrados temporalmente.
por_periodo = base.assign(
    coincide_programadas=base["diferencia_programadas"] == 0,
    coincide_entregadas=base["diferencia_entregadas"] == 0
).groupby("code_presentation").agg(
    matriculas=("id_student", "size"),
    estudiantes=("id_student", "nunique"),
    coincidencia_programadas=("coincide_programadas", "mean"),
    coincidencia_entregadas=("coincide_entregadas", "mean")
).reset_index()

resumen.to_csv(
    SALIDA / "154_auditoria_temporal_evaluaciones.csv", index=False
)
por_periodo.to_csv(
    SALIDA / "154_auditoria_por_presentacion.csv", index=False
)

print("\nEXPERIMENTO 154: AUDITORIA TEMPORAL DE PREDICTORES ACADEMICOS")
print("Corte temporal: dia", CORTE)
print("Matriculas en parquet V4:", len(features))
print("Duplicados por matricula:", features.duplicated(CLAVES).sum())

print("\nRECONSTRUCCION DESDE LOS CSV ORIGINALES")
print(resumen.to_string(index=False))

print("\nCOINCIDENCIA POR PRESENTACION")
print(por_periodo.round(5).to_string(index=False))

print("\nREGISTROS TEMPRANOS DE EVALUACIONES CON FECHA LIMITE FUTURA O AUSENTE")
print("Entregas <=34 con fecha limite >34:", len(tempranas_futuras))
print("Entregas <=34 con fecha limite ausente o >34:", len(registros_fuera))
print("De esas, no bancadas:", len(registros_fuera_no_banked))
print("Estas filas NO deben contar como evaluaciones entregadas hasta el dia 34.")

print("\nARCHIVOS GUARDADOS:")
print(SALIDA / "154_auditoria_temporal_evaluaciones.csv")
print(SALIDA / "154_auditoria_por_presentacion.csv")

if (base["diferencia_programadas"] != 0).any():
    raise AssertionError("No coincide el conteo de evaluaciones programadas.")
if (base["diferencia_entregadas"] != 0).any():
    raise AssertionError("No coincide el conteo de evaluaciones entregadas.")

print("\nRESULTADO: ambos predictores coinciden con la reconstruccion temporal estricta.")
