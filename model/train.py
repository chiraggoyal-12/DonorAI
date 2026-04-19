import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
df = pd.read_csv("../data/data.csv", sep=";")

# Target
df["y"] = df["y"].map({"yes": 1, "no": 0})

# Drop leakage column
df = df.drop(columns=["duration"])

# Split features and target
X = df.drop("y", axis=1)
y = df["y"]

categorical_cols = X.select_dtypes(include = ["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include = ["int64", "float64"]).columns.tolist()

print("Categorical:", categorical_cols)
print("Numerical:", numerical_cols)

numeric_transformer = Pipeline(steps = [
  ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps = [
  ("encoder", OneHotEncoder(drop = "first", handle_unknown = "ignore"))
])

preprocessor = ColumnTransformer(
  transformers = [
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
  ]
)

model = Pipeline(steps = [
  ("preprocessor", preprocessor),
  ("classifier", RandomForestClassifier(random_state = 42,
    n_estimators = 100,
    class_weight="balanced"
  ))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

baseline_model = model
baseline_model.fit(X_train, y_train)

baseline_probs = baseline_model.predict_proba(X_test)[:,1]
baseline_pred = (baseline_probs >= 0.3).astype(int)

print("\nBaseline Report:\n")
print(classification_report(y_test, baseline_pred))

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_split": [2, 5],
}

grid = GridSearchCV(
    model,                  # your pipeline
    param_grid,
    cv=3,
    scoring="f1",       # focus on donor detection
    n_jobs=-1,
    verbose=2
)

# Train with GridSearch
grid.fit(X_train, y_train)

# Replace model with best one
model = grid.best_estimator_

print("Best Params:", grid.best_params_)

y_probs = model.predict_proba(X_test)[:,1]
thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    score = f1_score(y_test, y_pred)
    
    if score > best_f1:
        best_f1 = score
        best_threshold = t

print("Best Threshold:", best_threshold)
print("Best F1:", best_f1)
y_pred_custom = (y_probs >= best_threshold).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred_custom))


import joblib

joblib.dump(model, "model.pkl")
