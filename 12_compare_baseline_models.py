
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "interim" /
    "fixed_week4_dataset.parquet"
)

OUTPUT_FILE = (
    BASE_DIR / "outputs" / "tables" /
    "model_comparison.csv"
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

# Separar entrenamiento y prueba por estudiante
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]
y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# Comparar cuatro alternativas
models = {
    "Baseline mayoritaria": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="most_frequent"))
    ]),

    "Regresión logística": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest completo": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Random Forest sin weekly_change": Pipeline([
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

for name, model in models.items():
    print("\nEntrenando:", name)

    if name == "Random Forest sin weekly_change":
        selected_features = [
            col for col in features if col != "weekly_change"
        ]
        X_train_model = X_train[selected_features]
        X_test_model = X_test[selected_features]
    else:
        X_train_model = X_train
        X_test_model = X_test

    model.fit(X_train_model, y_train)

    predictions = model.predict(X_test_model)

    # ROC-AUC no se calcula para el clasificador mayoritario,
    # porque sus probabilidades constantes no permiten comparar
    # la capacidad de ordenar las dos clases.
    if name == "Baseline mayoritaria":
        auc = None
    else:
        probabilities = model.predict_proba(X_test_model)[:, 1]
        auc = roc_auc_score(y_test, probabilities)

    results.append({
        "modelo": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": auc
    })

comparison = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_FILE, index=False)

print("\nCOMPARACIÓN DE MODELOS")
print(comparison.round(4).to_string(index=False))

print("\nArchivo guardado en:")
print(OUTPUT_FILE)
