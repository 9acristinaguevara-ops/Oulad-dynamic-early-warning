
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
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

METRICS_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "corrected_model_comparison.csv"
)

REPORT_FILE = (
    BASE_DIR / "outputs" / "logs" /
    "corrected_model_report.txt"
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

# Dividir por estudiante, no por filas aleatorias.

# Primera separacion: entrenamiento y datos temporales
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_index, temp_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_temp = X.iloc[temp_index]

y_train = y.iloc[train_index]
y_temp = y.iloc[temp_index]

groups_train = groups.iloc[train_index]
groups_temp = groups.iloc[temp_index]

# Segunda separacion: validacion y prueba
splitter_temp = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_index, test_index = next(
    splitter_temp.split(
        X_temp,
        y_temp,
        groups=groups_temp
    )
)

X_val = X_temp.iloc[val_index]
X_test = X_temp.iloc[test_index]

y_val = y_temp.iloc[val_index]
y_test = y_temp.iloc[test_index]

groups_val = groups_temp.iloc[val_index]
groups_test = groups_temp.iloc[test_index]

print("\nRegistros de entrenamiento:", len(X_train))
print("Registros de validacion:", len(X_val))
print("Registros de prueba:", len(X_test))

print(
    "Estudiantes compartidos entre entrenamiento y validacion:",
    len(set(groups_train) & set(groups_val))
)

print(
    "Estudiantes compartidos entre entrenamiento y prueba:",
    len(set(groups_train) & set(groups_test))
)

print(
    "Estudiantes compartidos entre validacion y prueba:",
    len(set(groups_val) & set(groups_test))
)


models = {
    "Referencia mayoritaria": DummyClassifier(
        strategy="most_frequent"
    ),

    "Regresion logistica": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results = []
report_lines = []


for name, model in models.items():
    print("\nEntrenando:", name)

    model.fit(X_train, y_train)

    # Seleccionar el umbral usando validacion
    if name == "Regresion logistica":
        val_probabilities = model.predict_proba(X_val)[:, 1]

        best_threshold = 0.50
        best_f1 = -1

        print("\nEvaluacion de umbrales en validacion:")

        for threshold in [0.30, 0.40, 0.50, 0.60]:
            val_predictions = (
                val_probabilities >= threshold
            ).astype(int)

            val_precision = precision_score(
                y_val, val_predictions, zero_division=0
            )
            val_recall = recall_score(
                y_val, val_predictions, zero_division=0
            )
            val_f1 = f1_score(
                y_val, val_predictions, zero_division=0
            )

            print(
                f"Umbral: {threshold:.2f} | "
                f"Precision: {val_precision:.4f} | "
                f"Recall: {val_recall:.4f} | "
                f"F1: {val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = threshold

        print("\nMejor umbral en validacion:", best_threshold)
        print("Mejor F1 en validacion:", round(best_f1, 4))

    # Evaluacion final con el conjunto de prueba
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    else:
        auc = float("nan")

    if name == "Regresion logistica":
        predictions = (
            probabilities >= best_threshold
        ).astype(int)
    else:
        predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )


    results.append({
        "modelo": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })

    report_lines.append(
        f"\nModelo: {name}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1: {f1:.4f}\n"
        f"ROC-AUC: {auc:.4f}\n"
        f"Matriz de confusion:\n{matrix}\n"
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Matriz de confusion:")
    print(matrix)

results_df = pd.DataFrame(results)

METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(METRICS_FILE, index=False)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("COMPARACION DE MODELOS CORREGIDOS\n")
    file.write(
        f"Entrenamiento: {len(X_train)} registros\n"
    )
    file.write(f"Prueba: {len(X_test)} registros\n")
    file.write(
        "Estudiantes compartidos: "
        f"{len(set(groups_train) & set(groups_test))}\n"
    )
    file.write("\n")
    file.write("\n".join(report_lines))

print("\nCOMPARACION FINAL")
print(results_df.to_string(index=False))

print("\nResultados guardados en:", METRICS_FILE)
print("Informe guardado en:", REPORT_FILE)
