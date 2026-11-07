
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

# 3. Calcular media, desviación estándar y CV
z["media_actividad"] = z[columnas].mean(axis=1)
z["desviacion"] = z[columnas].std(axis=1)

z["cv"] = z["desviacion"] / z["media_actividad"]

# 4. Identificar el percentil 99 del CV
limite = z["cv"].quantile(0.99)

print("Percentil 99 del CV:", round(limite, 5))
print("Registros originales:", len(z))

# 5. Excluir el 1% superior
z = z[z["cv"] <= limite].copy()

print("Registros después del filtro:", len(z))

# 6. Estandarizar variables
z["media_z"] = (
    z["media_actividad"] - z["media_actividad"].mean()
) / z["media_actividad"].std()

z["cv_z"] = (
    z["cv"] - z["cv"].mean()
) / z["cv"].std()

z["semanas_z"] = (
    z["semanas_observadas"] - z["semanas_observadas"].mean()
) / z["semanas_observadas"].std()

# 7. Crear interacción
z["interaccion_cv"] = z["cv_z"] * z["semanas_z"]

# 8. Ajustar modelo
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

# 9. Mostrar resultados
print("\nMODELO CV SIN EL 1% SUPERIOR")
print("============================")
print("Pseudo R2:", round(modelo.prsquared, 6))
print(modelo.summary2().tables[1].round(5))

print("\nPRUEBA DE INTERACCION")
print(modelo.wald_test("interaccion_cv = 0"))

print("\nANALISIS FINALIZADO")
