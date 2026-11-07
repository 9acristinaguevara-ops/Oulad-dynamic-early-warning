
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import norm

# 1. Cargar datos
x = pd.read_parquet(
    "data/interim/weekly_panel_weeks0_4.parquet"
)

columnas = [
    "clics_semana_" + str(i)
    for i in range(5)
]

# 2. Seleccionar registros con al menos 2 semanas
x["semanas_observadas"] = x[columnas].notna().sum(axis=1)
z = x[x["semanas_observadas"] >= 2].copy()

# 3. Calcular media y CV
z["media_actividad"] = z[columnas].mean(axis=1)
z["desviacion"] = z[columnas].std(axis=1)

z["cv"] = z["desviacion"] / z["media_actividad"]

# 4. Estandarizar variables
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

z["interaccion_cv"] = z["cv_z"] * z["semanas_z"]

# 5. Ajustar modelo
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

cov = modelo.cov_params()

media_semanas = z["semanas_observadas"].mean()
desv_semanas = z["semanas_observadas"].std()

# 6. Efectos condicionales
print("\nEFECTOS CONDICIONALES DEL CV")
print("============================")

for n in [2, 3, 4, 5]:

    valor = (n - media_semanas) / desv_semanas

    coef = (
        modelo.params["cv_z"]
        + modelo.params["interaccion_cv"] * valor
    )

    varianza = (
        cov.loc["cv_z", "cv_z"]
        + 2 * valor * cov.loc["cv_z", "interaccion_cv"]
        + valor**2 * cov.loc[
            "interaccion_cv", "interaccion_cv"
        ]
    )

    error = np.sqrt(varianza)
    estadistico = coef / error
    p = 2 * norm.sf(abs(estadistico))

    inferior = coef - 1.96 * error
    superior = coef + 1.96 * error

    print("\nSemanas:", n)
    print("Coeficiente:", round(coef, 5))
    print("Error estándar:", round(error, 5))
    print("Valor p:", round(p, 6))
    print("OR:", round(np.exp(coef), 5))
    print(
        "IC 95% OR:",
        round(np.exp(inferior), 5),
        "a",
        round(np.exp(superior), 5)
    )

print("\nANALISIS FINALIZADO")
