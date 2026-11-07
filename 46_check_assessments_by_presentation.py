
import pandas as pd

assessments = pd.read_csv("data/raw/csv/assessments.csv")

assessments["date"] = pd.to_numeric(
    assessments["date"], errors="coerce"
)

# Evaluaciones con fecha programada hasta el día 34
programadas = assessments[
    assessments["date"].notna() &
    (assessments["date"] <= 34)
]

# Contar evaluaciones por módulo y presentación
conteo = programadas.groupby(
    ["code_module", "code_presentation"]
)["id_assessment"].count().reset_index()

conteo = conteo.rename(
    columns={"id_assessment": "evaluaciones_hasta_34"}
)

print("EVALUACIONES POR MÓDULO Y PRESENTACIÓN")
print("----------------------------------------")

print(conteo.to_string(index=False))

print("\nTOTAL DE EVALUACIONES:", len(programadas))
print(
    "COMBINACIONES CON EVALUACIONES:",
    len(conteo)
)
