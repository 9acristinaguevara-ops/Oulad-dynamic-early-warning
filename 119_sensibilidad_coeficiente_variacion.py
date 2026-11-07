
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import norm

# 1. Cargar los datos
x = pd.read_parquet(
    "data/interim/weekly_panel_weeks0_4.parquet"
)

columnas = [
    "clics_semana_" + str(i)
    for i in range(5)
]

# 2. Calcular semanas observadas
x["semanas_observadas"] = (
    x[columnas].notna().sum(axis=1)
)

# 3. Seleccionar estudiantes con al menos 2 semanas
z = x[x["semanas_observadas"] >= 2].copy()

# 4. Calcular media, desviación estándar y CV
z["media_actividad"] = z[columnas].mean(axis=1)
z["desviacion_actividad"] = z[columnas].std(axis=1)

z["cv"] = (
    z["desviacion_actividad"] /
    z["media_actividad"]
)

# 5. Estandarizar variables
z["media_z"] = (
    z["media_actividad"] - z["media_actividad"].mean()
) / z["media_actividad"].std()

z["cv_z"] = (
    z["cv"] - z["cv"].mean()
) / z["cv"].std()

z["semanas_z"] = (
    z["semanas_observadas"] -
    z["semanas_observadas"].mean()
) / z["semanas_observadas"].std()

# 6. Crear interacción
z["interaccion_cv"] = z["cv_z"] * z["semanas_z"]

# 7. Ajustar modelo logístico
variables = [
    "media_z",
    "cv_z",
    "semanas_z",
    "interaccion_cv"
]

modelo = sm.Logit(
    z["target_future_withdrawal"],
    sm.add_constant(z[variables])
).fit(disp=0, cov_type="HC3")

# 8. Mostrar resultados
print("\nMODELO DE SENSIBILIDAD CON CV")
print("================================")
print("Estudiantes:", len(z))
print("Pseudo R2:", round(modelo.prsquared, 6))

print("\nCOEFICIENTES DEL MODELO")
print(modelo.summary2().tables[1].round(5))

# 9. Evaluar la interacción
print("\nPRUEBA DE INTERACCION")
print(modelo.wald_test("interaccion_cv = 0"))

print("\nANALISIS FINALIZADO")
