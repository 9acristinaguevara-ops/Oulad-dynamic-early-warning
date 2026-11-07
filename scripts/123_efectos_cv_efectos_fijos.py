
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

# 2. Seleccionar estudiantes con al menos 2 semanas
x["semanas_observadas"] = x[columnas].notna().sum(axis=1)
z = x[x["semanas_observadas"] >= 2].copy()

# 3. Calcular media, desviación y CV
z["media_actividad"] = z[columnas].mean(axis=1)
z["desviacion"] = z[columnas].std(axis=1)
z["cv"] = z["desviacion"] / z["media_actividad"]

# 4. Estandarizar variables
for variable in [
    "media_actividad",
    "cv",
    "semanas_observadas"
]:
    z[variable + "_z"] = (
        z[variable] - z[variable].mean()
    ) / z[variable].std()

# 5. Crear interacción
z["interaccion_cv"] = (
    z["cv_z"] * z["semanas_observadas_z"]
)

# 6. Crear grupos académicos
z["grupo"] = (
    z["code_module"].astype(str)
    + "_"
    + z["code_presentation"].astype(str)
)

grupos = pd.get_dummies(
    z["grupo"],
    prefix="grupo",
    drop_first=True,
    dtype=float
)

# 7. Construir modelo
variables = [
    "media_actividad_z",
    "cv_z",
    "semanas_observadas_z",
    "interaccion_cv"
]

X = pd.concat([z[variables], grupos], axis=1)
X = sm.add_constant(X)

modelo = sm.Logit(
    z["target_future_withdrawal"],
    X
).fit(disp=0, cov_type="HC3")

# 8. Calcular efectos condicionales
cov = modelo.cov_params()

media_semanas = z["semanas_observadas"].mean()
desv_semanas = z["semanas_observadas"].std()

print("\nEFECTOS CONDICIONALES DEL CV")
print("MODELO CON EFECTOS FIJOS")
print("============================")

for n in [2, 3, 4, 5]:

    valor = (
        n - media_semanas
    ) / desv_semanas

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
