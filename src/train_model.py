"""Trains a small model (stand-in for the Exp 5/6 model) and saves the full pipeline.
Also creates the batch input file (with a few deliberately bad rows)."""
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
MODEL_VERSION = "v1.0"

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target          # target: 0 = malignant, 1 = benign
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
metrics = {"accuracy": round(accuracy_score(y_test, pred), 4),
           "f1": round(f1_score(y_test, pred), 4)}

joblib.dump({"pipeline": pipe,
             "version": MODEL_VERSION,
             "features": list(X.columns),
             "class_names": {0: "malignant", 1: "benign"}},
            "models/model.joblib")
json.dump({"version": MODEL_VERSION, **metrics}, open("models/model_info.json", "w"), indent=2)
print("Model trained:", metrics)

# ---- Batch input: 1000 records built from unseen test rows, plus a few bad rows ----
batch = pd.concat([X_test] * 9, ignore_index=True).iloc[:1000].copy()
batch.insert(0, "record_id", [f"REC{str(i+1).zfill(4)}" for i in range(len(batch))])

bad = batch.copy()
bad["mean area"] = bad["mean area"].astype(object)
bad.loc[5, "mean radius"] = None                 # missing value -> imputed by pipeline
bad.loc[10, "mean area"] = "abc"                 # invalid data type
bad.loc[20, "mean texture"] = -5                 # invalid (negative) value
bad = pd.concat([bad, bad.iloc[[30]]], ignore_index=True)   # exact duplicate row
bad.to_csv("data/batch_input.csv", index=False)
print("Batch input rows:", len(bad))
