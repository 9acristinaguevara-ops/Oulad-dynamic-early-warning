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
x["semanas_observadas"] = x[columnas].notna().sum(axis=1)

# 3. Seleccionar estudiantes con al menos 2 semanas
z = x[x["semanas_observadas"] >= 2].copy()

# 4. Calcular media y desviación estándar
a = z[columnas].mean(axis=1)
b = z[columnas].std(axis=1)

z["media"] = (a - a.mean()) / a.std()
z["sd"] = (b - b.mean()) / b.std()

# 5. Estandarizar cantidad de semanas
z["semanas"] = (
    z["semanas_observadas"] - z["semanas_observadas"].mean()
) / z["semanas_observadas"].std()

# 6. Crear interacción
z["interaccion"] = z["sd"] * z["semanas"]

# 7. Ajustar modelo logístico
variables = ["media", "sd", "semanas", "interaccion"]

modelo = sm.Logit(
    z["target_future_withdrawal"],
    sm.add_constant(z[variables])
).fit(disp=0, cov_type="HC3")

# 8. Calcular efectos condicionales
cov = modelo.cov_params()

media_semanas = z["semanas_observadas"].mean()
desv_semanas = z["semanas_observadas"].std()

print("\nEFECTO DE LA VARIABILIDAD POR SEMANAS")
print("======================================")

for n in [2, 3, 4, 5]:

    valor = (n - media_semanas) / desv_semanas

    coeficiente = (
        modelo.params["sd"]
        + modelo.params["interaccion"] * valor
    )

    varianza = (
        cov.loc["sd", "sd"]
        + 2 * valor * cov.loc["sd", "interaccion"]
        + valor**2 * cov.loc["interaccion", "interaccion"]
    )

    error = np.sqrt(varianza)

    estadistico = coeficiente / error

    p_valor = 2 * norm.sf(abs(estadistico))

    limite_inferior = coeficiente - 1.96 * error
    limite_superior = coeficiente + 1.96 * error

    odds_ratio = np.exp(coeficiente)

    print("\nSemanas observadas:", n)
    print("Coeficiente:", round(coeficiente, 5))
    print("Error estándar:", round(error, 5))
    print("Valor p:", round(p_valor, 6))
    print("OR:", round(odds_ratio, 5))
    print(
        "IC 95%:",
        round(limite_inferior, 5),
        "a",
        round(limite_superior, 5)
    )

print("\nANÁLISIS FINALIZADO")