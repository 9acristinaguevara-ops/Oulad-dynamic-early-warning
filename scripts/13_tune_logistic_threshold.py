
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "interim" /
    "fixed_week4_dataset.parquet"
)

OUTPUT_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "logistic_threshold_comparison.csv"
)

print("Cargando datos...")
df = pd.read_parquet(INPUT_FILE)

features = [
    "weekly_clicks",
    "active_days",
    "activity_records",
    "has_previous_week",
    "weekly_change",
    "activity_decreased"
]

features = [col for col in features if col in df.columns]

X = df[features].copy()
y = df["target_withdrawn"].copy()
groups = df["id_student"].copy()

# Primera separación: conjunto de prueba independiente
splitter_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter_test.split(X, y, groups=groups)
)

X_train_all = X.iloc[train_index]
y_train_all = y.iloc[train_index]
groups_train = groups.iloc[train_index]

X_test = X.iloc[test_index]
y_test = y.iloc[test_index]

# Segunda separación: entrenamiento y validación
splitter_val = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=21
)

inner_train_index, val_index = next(
    splitter_val.split(
        X_train_all,
        y_train_all,
        groups=groups_train
    )
)

X_train = X_train_all.iloc[inner_train_index]
y_train = y_train_all.iloc[inner_train_index]

X_val = X_train_all.iloc[val_index]
y_val = y_train_all.iloc[val_index]

# Entrenar usando únicamente el conjunto de entrenamiento interno
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

print("Entrenando regresión logística...")
model.fit(X_train, y_train)

# Usar validación para comparar umbrales
val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
results = []

for threshold in thresholds:
    predictions = (val_probabilities >= threshold).astype(int)

    results.append({
        "umbral": threshold,
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

comparison = pd.DataFrame(results)

# Elegir el umbral con mayor F1 en validación
best_index = comparison["f1"].idxmax()
best_threshold = comparison.loc[best_index, "umbral"]

print("\nRESULTADOS EN VALIDACIÓN")
print(comparison.round(4).to_string(index=False))
print("\nUmbral elegido por F1:", best_threshold)

# Reentrenar con todo el conjunto de entrenamiento original
final_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

final_model.fit(X_train_all, y_train_all)

# Evaluar una sola vez en el conjunto de prueba
test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nEVALUACIÓN FINAL EN PRUEBA")
print("Umbral utilizado:", best_threshold)
print("Precision:", round(
    precision_score(y_test, test_predictions, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, test_predictions, zero_division=0), 4
))
print("F1:", round(
    f1_score(y_test, test_predictions, zero_division=0), 4
))
print("ROC-AUC:", round(
    roc_auc_score(y_test, test_probabilities), 4
))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_FILE, index=False)

print("\nComparación guardada en:", OUTPUT_FILE)
