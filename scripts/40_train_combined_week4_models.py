
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Cargar los datos
archivo = "data/interim/week4_combined_features.parquet"
datos = pd.read_parquet(archivo)

# Variables que no deben utilizarse como predictores
excluir = [
    "code_module",
    "code_presentation",
    "id_student",
    "date_registration",
    "date_unregistration",
    "final_result",
    "target_future_withdrawal"
]

# Seleccionar variables predictoras
X = datos.drop(columns=excluir)
y = datos["target_future_withdrawal"]
grupos = datos["id_student"]

# Separar entrenamiento y prueba por estudiante
separador = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

indices_entrenamiento, indices_prueba = next(
    separador.split(X, y, groups=grupos)
)

X_train = X.iloc[indices_entrenamiento]
X_test = X.iloc[indices_prueba]
y_train = y.iloc[indices_entrenamiento]
y_test = y.iloc[indices_prueba]

print("TAMANO DE LOS DATOS")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))
print("Variables:", len(X.columns))

# Modelo 1: Regresion Logistica
modelo_logistico = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
    ("modelo", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

modelo_logistico.fit(X_train, y_train)

# Modelo 2: Random Forest
modelo_rf = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("modelo", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

modelo_rf.fit(X_train, y_train)

# Evaluar ambos modelos
modelos = [
    ("Regresion Logistica", modelo_logistico),
    ("Random Forest", modelo_rf)
]

for nombre, modelo in modelos:
    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 45)
    print(nombre)
    print("=" * 45)

    print("Accuracy:", round(
        accuracy_score(y_test, predicciones), 4
    ))
    print("Precision:", round(
        precision_score(y_test, predicciones, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_test, predicciones, zero_division=0), 4
    ))
    print("F1:", round(
        f1_score(y_test, predicciones, zero_division=0), 4
    ))
    print("ROC AUC:", round(
        roc_auc_score(y_test, probabilidades), 4
    ))

    print("Matriz de confusion:")
    print(confusion_matrix(y_test, predicciones))
