"""Batch prediction system: reads a CSV, validates, predicts all valid rows in ONE call,
stores results with timestamp + model version, and records performance."""
import argparse, json, logging, time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.FileHandler("outputs/batch_run.log", mode="w"),
                              logging.StreamHandler()])
log = logging.getLogger("batch")

def load_model(path):
    art = joblib.load(path)
    log.info("Loaded model version %s (%d features)", art["version"], len(art["features"]))
    return art

def validate(df, features):
    """Returns (valid_df, failed_df). Failed rows carry a reason; nothing is silently dropped."""
    failures = []
    missing_cols = [c for c in ["record_id"] + features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema error, missing columns: {missing_cols}")

    dup_mask = df.duplicated(keep="first")
    for idx in df[dup_mask].index:
        failures.append((idx, "duplicate record"))
    work = df[~dup_mask].copy()

    for c in features:
        coerced = pd.to_numeric(work[c], errors="coerce")
        bad = work[c].notna() & coerced.isna()
        for idx in work[bad].index:
            failures.append((idx, f"non-numeric value in '{c}'"))
        work[c] = coerced
    work = work.drop(index=[i for i, _ in failures if i in work.index])

    neg = (work[features] < 0).any(axis=1)
    for idx in work[neg].index:
        failures.append((idx, "negative value (invalid)"))
    work = work[~neg]

    n_missing = int(work[features].isna().any(axis=1).sum())
    if n_missing:
        log.warning("%d valid rows have missing values -> handled by pipeline imputer", n_missing)

    failed = pd.DataFrame(failures, columns=["row_index", "reason"])
    failed.insert(1, "record_id", df.loc[failed["row_index"], "record_id"].values if len(failed) else [])
    return work, failed

def main(args):
    t_start = time.perf_counter()
    art = load_model(args.model)
    pipe, features, names = art["pipeline"], art["features"], art["class_names"]

    df = pd.read_csv(args.input)
    total = len(df)
    log.info("Read %d input records from %s", total, args.input)

    valid, failed = validate(df, features)
    log.info("Validation: %d valid, %d failed", len(valid), len(failed))
    for _, r in failed.iterrows():
        log.error("Rejected row %s (%s): %s", r["row_index"], r["record_id"], r["reason"])

    t_pred = time.perf_counter()
    X = valid[features]
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)
    pred_time = time.perf_counter() - t_pred

    out = pd.DataFrame({
        "record_id": valid["record_id"].values,
        "predicted_class": preds,
        "predicted_label": [names[int(p)] for p in preds],
        "confidence": np.round(proba.max(axis=1), 4),
        "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": art["version"],
    })
    out.to_csv(args.output, index=False)
    failed.to_csv("outputs/failed_records.csv", index=False)

    total_time = time.perf_counter() - t_start
    ok = len(out)
    perf = {
        "total_input_records": total,
        "successful_predictions": ok,
        "failed_records": len(failed),
        "output_rows_match_successful": ok == total - len(failed),
        "total_time_sec": round(total_time, 4),
        "prediction_only_time_sec": round(pred_time, 4),
        "avg_time_per_record_ms": round(total_time / total * 1000, 4),
        "throughput_records_per_sec": round(ok / total_time, 1),
        "model_version": art["version"],
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(perf, open("outputs/performance.json", "w"), indent=2)
    log.info("Saved predictions -> %s", args.output)
    log.info("Performance: %s", perf)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/batch_input.csv")
    ap.add_argument("--model", default="models/model.joblib")
    ap.add_argument("--output", default="outputs/predictions.csv")
    try:
        main(ap.parse_args())
    except Exception:
        log.exception("Batch run failed")
        raise SystemExit(1)
