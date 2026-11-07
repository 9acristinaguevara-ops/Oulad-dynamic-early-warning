
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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

DATA_FILE = (
    BASE_DIR / "data" / "interim" /
    "corrected_week4_dataset.parquet"
)

OUTPUT_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "corrected_threshold_comparison.csv"
)

REPORT_FILE = (
    BASE_DIR / "outputs" / "logs" /
    "corrected_threshold_report.txt"
)

print("Cargando conjunto corregido...")
df = pd.read_parquet(DATA_FILE)

features = [
    "weekly_clicks",
    "active_days",
    "activity_records",
    "has_previous_week",
    "weekly_change",
    "activity_decreased"
]

X = df[features].copy()
y = df["target_future_withdrawal"]
groups = df["id_student"]

# Separar primero el conjunto de prueba.
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
groups_test = groups.iloc[test_index]

# Separar entrenamiento y validación por estudiante.
splitter_val = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=21
)

train_inner, val_inner = next(
    splitter_val.split(
        X_train_all,
        y_train_all,
        groups=groups_train
    )
)

X_train = X_train_all.iloc[train_inner]
y_train = y_train_all.iloc[train_inner]

X_val = X_train_all.iloc[val_inner]
y_val = y_train_all.iloc[val_inner]

groups_inner_train = groups_train.iloc[train_inner]
groups_val = groups_train.iloc[val_inner]

print("\nEntrenamiento:", len(X_train))
print("Validación:", len(X_val))
print("Prueba:", len(X_test))

print(
    "Estudiantes compartidos entre entrenamiento y validación:",
    len(set(groups_inner_train) & set(groups_val))
)

print(
    "Estudiantes compartidos entre entrenamiento y prueba:",
    len(set(groups_train) & set(groups_test))
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

print("\nEntrenando regresión logística...")
model.fit(X_train, y_train)

# Buscar el umbral usando únicamente validación.
prob_val = model.predict_proba(X_val)[:, 1]

thresholds = [
    0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70
]

results = []

for threshold in thresholds:
    predictions = (prob_val >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    results.append({
        "umbral": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

results_df = pd.DataFrame(results)

# Elegir el umbral con mayor F1 en validación.
best_index = results_df["f1"].idxmax()
best_threshold = results_df.loc[best_index, "umbral"]

print("\nRESULTADOS EN VALIDACION")
print(results_df.round(4).to_string(index=False))

print("\nUmbral seleccionado:", best_threshold)

# Evaluar una sola vez en el conjunto de prueba.
prob_test = model.predict_proba(X_test)[:, 1]
pred_test = (prob_test >= best_threshold).astype(int)

precision_test = precision_score(
    y_test, pred_test, zero_division=0
)

recall_test = recall_score(
    y_test, pred_test, zero_division=0
)

f1_test = f1_score(
    y_test, pred_test, zero_division=0
)

auc_test = roc_auc_score(y_test, prob_test)

matrix = confusion_matrix(
    y_test, pred_test, labels=[0, 1]
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(OUTPUT_FILE, index=False)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("AJUSTE DEL UMBRAL - CONJUNTO CORREGIDO\n\n")
    file.write(results_df.round(4).to_string(index=False))
    file.write(f"\n\nUmbral seleccionado: {best_threshold}\n")
    file.write(f"Precision en prueba: {precision_test:.4f}\n")
    file.write(f"Recall en prueba: {recall_test:.4f}\n")
    file.write(f"F1 en prueba: {f1_test:.4f}\n")
    file.write(f"ROC-AUC en prueba: {auc_test:.4f}\n")
    file.write(f"Matriz de confusion:\n{matrix}\n")

print("\nRESULTADOS FINALES EN PRUEBA")
print("Precision:", round(precision_test, 4))
print("Recall:", round(recall_test, 4))
print("F1:", round(f1_test, 4))
print("ROC-AUC:", round(auc_test, 4))
print("Matriz de confusion:")
print(matrix)

print("\nTabla guardada en:", OUTPUT_FILE)
print("Informe guardado en:", REPORT_FILE)
