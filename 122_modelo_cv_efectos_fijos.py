
import pandas as pd
import statsmodels.api as sm
import numpy as np

# 1. Cargar datos
x = pd.read_parquet(
    "data/interim/weekly_panel_weeks0_4.parquet"
)

columnas = [
    "clics_semana_" + str(i)
    for i in range(5)
]

# 2. Seleccionar estudiantes con al menos 2 semanas
x["semanas_observadas"] = x[columnas].notna().sum(axis=1)
z = x[x["semanas_observadas"] >= 2].copy()

# 3. Calcular actividad media, desviación y CV
z["media_actividad"] = z[columnas].mean(axis=1)
z["desviacion"] = z[columnas].std(axis=1)

z["cv"] = (
    z["desviacion"] / z["media_actividad"]
)

# 4. Estandarizar variables
for variable in ["media_actividad", "cv", "semanas_observadas"]:

    z[variable + "_z"] = (
        z[variable] - z[variable].mean()
    ) / z[variable].std()

# 5. Crear interacción
z["interaccion_cv"] = (
    z["cv_z"] * z["semanas_observadas_z"]
)

# 6. Identificar combinación de módulo y presentación
z["grupo"] = (
    z["code_module"].astype(str)
    + "_"
    + z["code_presentation"].astype(str)
)

# 7. Crear variables indicadoras de los grupos
grupos = pd.get_dummies(
    z["grupo"],
    prefix="grupo",
    drop_first=True,
    dtype=float
)

# 8. Construir matriz de variables
variables = z[
    [
        "media_actividad_z",
        "cv_z",
        "semanas_observadas_z",
        "interaccion_cv"
    ]
]

X = pd.concat([variables, grupos], axis=1)

X = sm.add_constant(X)

# 9. Ajustar regresión logística
modelo = sm.Logit(
    z["target_future_withdrawal"],
    X
).fit(disp=0, cov_type="HC3")

# 10. Mostrar resultados principales
print("\nMODELO CV CON EFECTOS FIJOS")
print("============================")

print("Estudiantes:", len(z))
print("Grupos académicos:", z["grupo"].nunique())
print("Pseudo R2:", round(modelo.prsquared, 6))

print("\nCOEFICIENTES PRINCIPALES")

print(
    modelo.summary2().tables[1]
    .loc[
        [
            "media_actividad_z",
            "cv_z",
            "semanas_observadas_z",
            "interaccion_cv"
        ]
    ]
    .round(5)
)

print("\nPRUEBA DE INTERACCION")

print(
    modelo.wald_test("interaccion_cv = 0")
)

print("\nANALISIS FINALIZADO")
