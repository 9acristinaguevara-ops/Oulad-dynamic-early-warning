
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

datos = pd.read_parquet(
    "data/interim/week4_enriched_features.parquet"
)

variables = [
    "week3_clicks",
    "week4_clicks",
    "week4_active_days",
    "week4_activity_records",
    "has_week4_activity",
    "weekly_change",
    "semanas_con_actividad",
    "clics_acumulados_0_4",
    "promedio_clics_0_4",
    "dias_activos_0_4",
    "clics_recientes_3_4",
    "clics_anteriores_0_2"
]

X = datos[variables]
y = datos["target_future_withdrawal"]
grupos = datos["id_student"]

separador = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

indices_train, indices_test = next(
    separador.split(X, y, groups=grupos)
)

X_train = X.iloc[indices_train]
X_test = X.iloc[indices_test]
y_train = y.iloc[indices_train]
y_test = y.iloc[indices_test]

modelo = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
    ("modelo", LogisticRegression(max_iter=1000))
])

modelo.fit(X_train, y_train)

probabilidades = modelo.predict_proba(X_test)[:, 1]

print("PROBABILIDADES DE RETIRADA FUTURA")
print(pd.Series(probabilidades).describe().round(4))

print("\nPROBABILIDAD MEDIA POR CLASE REAL")
print(
    pd.DataFrame({
        "clase_real": y_test.to_numpy(),
        "probabilidad": probabilidades
    }).groupby("clase_real")["probabilidad"].mean().round(4)
)

print("\nPREDICCIONES CON DIFERENTES UMBRALES")

for umbral in [0.10, 0.15, 0.20, 0.30, 0.50]:
    predicciones = (probabilidades >= umbral).astype(int)

    verdaderos_positivos = (
        (predicciones == 1) & (y_test.to_numpy() == 1)
    ).sum()

    positivos_detectados = predicciones.sum()

    print(
        "Umbral:", umbral,
        "| Alertas:", positivos_detectados,
        "| Retiradas detectadas:", verdaderos_positivos
    )
