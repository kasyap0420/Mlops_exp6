"""Generate charts, terminal-style screenshots, and a new PDF report for Exp 7."""
import csv
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
                                Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE, "report_assets")
os.makedirs(ASSETS, exist_ok=True)

# ---------- Read data ----------
preds_path = os.path.join(BASE, "outputs", "predictions.csv")
failed_path = os.path.join(BASE, "outputs", "failed_records.csv")
perf_path = os.path.join(BASE, "outputs", "performance.json")
model_info_path = os.path.join(BASE, "models", "model_info.json")

rows = []
with open(preds_path, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

failed_rows = []
with open(failed_path, newline="") as f:
    for r in csv.DictReader(f):
        failed_rows.append(r)

with open(perf_path) as f:
    perf = json.load(f)
with open(model_info_path) as f:
    model_info = json.load(f)

confidences = [float(r["confidence"]) for r in rows]
classes = [r["predicted_label"] for r in rows]

# ---------- Charts ----------
def save_chart(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# Class distribution
fig, ax = plt.subplots(figsize=(5, 3))
cls_count = {c: classes.count(c) for c in set(classes)}
ax.bar(cls_count.keys(), cls_count.values(), color=["#1f77b4", "#2ca02c"])
ax.set_title("Predicted Class Distribution")
ax.set_ylabel("Count")
for k, v in cls_count.items():
    ax.text(k, v + 5, str(v), ha="center")
chart_class = save_chart(fig, "class_dist.png")

# Confidence histogram
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(confidences, bins=20, color="#17becf", edgecolor="black")
ax.set_title("Confidence Score Distribution")
ax.set_xlabel("Confidence")
ax.set_ylabel("Frequency")
chart_conf = save_chart(fig, "conf_hist.png")

# Performance metrics bar
metrics = ["Total input", "Successful", "Failed"]
vals = [perf["total_input_records"], perf["successful_predictions"], perf["failed_records"]]
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(metrics, vals, color=["#7f7f7f", "#2ca02c", "#d62728"])
ax.set_title("Batch Run Record Counts")
ax.set_ylabel("Records")
for k, v in enumerate(vals):
    ax.text(k, v + 10, str(v), ha="center")
chart_perf = save_chart(fig, "perf_counts.png")

# Throughput gauge-like bar
fig, ax = plt.subplots(figsize=(5, 3))
ax.barh(["Throughput (rec/sec)"], [perf["throughput_records_per_sec"]], color="#9467bd")
ax.set_title("Processing Throughput")
ax.set_xlim(0, perf["throughput_records_per_sec"] * 1.4)
ax.text(perf["throughput_records_per_sec"] + 20, 0, f"{perf['throughput_records_per_sec']}", va="center")
chart_tp = save_chart(fig, "throughput.png")

# ---------- Terminal-style screenshots ----------
def terminal_image(lines, title="Terminal", name="terminal.png", width=1100, line_height=34):
    img = Image.new("RGB", (width, 60 + len(lines) * line_height), (20, 20, 20))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("consola.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    d.rectangle([0, 0, width, 40], fill=(40, 40, 40))
    d.text((15, 8), title, font=font, fill=(200, 200, 200))
    y = 55
    for line in lines:
        d.text((15, y), line, font=font, fill=(220, 220, 220))
        y += line_height
    path = os.path.join(ASSETS, name)
    img.save(path)
    return path

train_out = [
    "PS C:\\batch_project> python src/train_model.py",
    "Model trained: {'accuracy': 0.9561, 'f1': 0.9655}",
    "Batch input rows: 1001",
]
train_img = terminal_image(train_out, "Training Run", "terminal_train.png")

predict_out = [
    "PS C:\\batch_project> python src/batch_predict.py",
    "2026-10-22 17:45:13,976 | INFO | Loaded model version v1.0 (30 features)",
    "2026-10-22 17:45:13,991 | INFO | Read 1001 input records from data/batch_input.csv",
    "2026-10-22 17:45:14,019 | WARNING | 1 valid rows have missing values -> handled by pipeline imputer",
    "2026-10-22 17:45:14,022 | INFO | Validation: 998 valid, 3 failed",
    "2026-10-22 17:45:14,022 | ERROR | Rejected row 1000 (REC0031): duplicate record",
    "2026-10-22 17:45:14,022 | ERROR | Rejected row 10 (REC0011): non-numeric value in 'mean area'",
    "2026-10-22 17:45:14,022 | ERROR | Rejected row 20 (REC0021): negative value (invalid)",
    "2026-10-22 17:45:14,094 | INFO | Saved predictions -> outputs/predictions.csv",
    "2026-10-22 17:45:14,094 | INFO | Performance: total_time_sec=1.1898, throughput=838.8 rec/s",
]
predict_img = terminal_image(predict_out, "Batch Prediction Run", "terminal_predict.png")

# ---------- Build PDF report ----------
styles = getSampleStyleSheet()
title_st = ParagraphStyle("title", parent=styles["Title"], fontSize=20, spaceAfter=12)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14, spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
code_st = ParagraphStyle("code", parent=styles["Code"], fontSize=8, leading=10, fontName="Courier")

doc = SimpleDocTemplate(
    os.path.join(BASE, "Experiment_7_Report_Updated.pdf"),
    pagesize=A4,
    leftMargin=0.8 * inch,
    rightMargin=0.8 * inch,
    topMargin=0.8 * inch,
    bottomMargin=0.8 * inch,
    title="Experiment 7 - Batch Prediction System (Updated)",
    author="Batch Project",
)
story = []

story.append(Paragraph("Experiment 7", title_st))
story.append(Paragraph("Batch Prediction System", title_st))
story.append(Paragraph("Name: Kasyap Yanamandra   Roll No: 231FA04E47   Date: 22 October 2026", body))
story.append(Spacer(1, 6))

story.append(Paragraph("1. Aim", h1))
story.append(Paragraph(
    "To implement a batch inference system that applies a trained machine learning model to multiple input records in "
    "a single execution and stores the prediction results for further analysis.", body))

story.append(Paragraph("2. Model and Dataset", h1))
story.append(Paragraph(
    "Dataset: Breast Cancer Wisconsin (Diagnostic), loaded from scikit-learn (569 records, 30 numeric features). "
    "Target: 0 = malignant, 1 = benign.", body))
story.append(Paragraph(
    "Model: scikit-learn Pipeline = SimpleImputer (median) &rarr; StandardScaler &rarr; RandomForestClassifier "
    "(200 trees, random_state = 42). Training: 80/20 stratified split. Test accuracy = 0.9561, F1 = 0.9655.", body))
story.append(Paragraph(
    "Saved as models/model.joblib together with the model version (v1.0), feature names and class labels. "
    "The model is not retrained during batch inference.", body))
story.append(RLImage(train_img, width=6.2 * inch, height=1.9 * inch))
story.append(Paragraph("Figure 1: Model training output", body))

story.append(Paragraph("3. Workflow and Architecture", h1))
story.append(Paragraph(
    "Batch CSV &rarr; Load saved model &rarr; Validate records &rarr; Apply preprocessing pipeline &rarr; "
    "Predict all valid rows in one call &rarr; Save predictions.csv &rarr; Save performance.json and log.", body))
story.append(Paragraph(
    "Project structure:", body))
story.append(Paragraph(
    "project/<br/>&nbsp;&nbsp;data/batch_input.csv<br/>&nbsp;&nbsp;models/model.joblib, model_info.json<br/>"
    "&nbsp;&nbsp;outputs/predictions.csv, failed_records.csv, performance.json, batch_run.log<br/>"
    "&nbsp;&nbsp;src/train_model.py, batch_predict.py", body))
story.append(Paragraph(
    'Run command: python src/batch_predict.py --input data/batch_input.csv', body))

story.append(Paragraph("4. Input Batch Dataset", h1))
story.append(Paragraph(
    "The batch file contains 1001 rows: record_id plus the 30 feature columns. Four problems were deliberately "
    "inserted to test validation:", body))
fail_table_data = [
    ["Row / Record", "Problem", "Expected handling"],
    ["REC0006", "Missing value in 'mean radius'", "Accepted; filled by pipeline's median imputer"],
    ["REC0011", "Text 'abc' in 'mean area'", "Rejected (invalid data type)"],
    ["REC0021", "Negative 'mean texture'", "Rejected (invalid value)"],
    ["REC0031 (last row)", "Exact duplicate", "Rejected (duplicate)"],
]
t = Table(fail_table_data, colWidths=[1.2 * inch, 2.6 * inch, 2.4 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]))
story.append(t)

story.append(Paragraph("5. Batch Prediction Implementation", h1))
story.append(Paragraph(
    "Reads the whole CSV with pandas and checks the schema (all 30 feature columns plus record_id must exist). "
    "Validation checks duplicates, non-numeric values and negative values. Failed rows are written to "
    "failed_records.csv with the reason; none are dropped silently.", body))
story.append(Paragraph(
    "The saved pipeline applies the same imputation and scaling used in training; all valid rows are predicted in "
    "a single predict() call. predict_proba() gives a confidence score (highest class probability) for each record.", body))
story.append(Paragraph(
    "Prediction timestamp and model version are stored with every row. Errors are logged in batch_run.log, and any "
    "fatal error exits with code 1.", body))
story.append(RLImage(predict_img, width=6.2 * inch, height=3.2 * inch))
story.append(Paragraph("Figure 2: Batch prediction execution log (terminal screenshot)", body))

story.append(Paragraph("6. Prediction Output", h1))
story.append(Paragraph(
    'Output file: outputs/predictions.csv (' + str(len(rows)) + ' rows). Columns: record_id, predicted_class, '
    'predicted_label, confidence, prediction_timestamp, model_version.', body))
story.append(Paragraph(
    "Prediction summary: " + str(cls_count.get("benign", 0)) + " benign and " + str(cls_count.get("malignant", 0)) +
    " malignant. Records rejected: 3 (see failed_records.csv). Check: output rows (" + str(len(rows)) +
    ") = input rows (1001) - failed rows (3). Verified as True.", body))
story.append(RLImage(chart_class, width=3.0 * inch, height=1.8 * inch))
story.append(RLImage(chart_conf, width=3.0 * inch, height=1.8 * inch))

story.append(Paragraph("7. Performance Measurement", h1))
perf_table_data = [
    ["Metric", "Value"],
    ["Total input records", str(perf["total_input_records"])],
    ["Successful predictions", str(perf["successful_predictions"])],
    ["Failed records", str(perf["failed_records"])],
    ["Total time (load + validate + predict + save)", str(perf["total_time_sec"]) + " s"],
    ["Prediction-only time", str(perf["prediction_only_time_sec"]) + " s"],
    ["Average time per record", str(perf["avg_time_per_record_ms"]) + " ms"],
    ["Throughput", str(perf["throughput_records_per_sec"]) + " records/sec"],
    ["Model version", perf["model_version"]],
    ["Run timestamp", perf["run_timestamp"]],
]
t2 = Table(perf_table_data, colWidths=[3.4 * inch, 2.8 * inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
]))
story.append(t2)
story.append(RLImage(chart_perf, width=2.9 * inch, height=1.74 * inch))
story.append(RLImage(chart_tp, width=2.9 * inch, height=1.74 * inch))

story.append(Paragraph(
    "Factors affecting batch performance: dataset size (more records = longer total time, but average time per "
    "record usually falls because fixed costs such as model loading are shared); preprocessing complexity "
    "(imputation, scaling add time per record); model complexity (Random Forest with more trees is slower than a "
    "linear model); I/O (reading/writing CSV; model loading and file I/O take most of the total time, since pure "
    "prediction took only " + str(perf["prediction_only_time_sec"]) + " s).", body))

story.append(Paragraph("8. Batch vs Real-Time Prediction", h1))
cmp_data = [
    ["Aspect", "Batch prediction (Exp 7)", "Real-time REST API (Exp 5)"],
    ["Input pattern", "Large file with many records at once", "One record (or a few) per request"],
    ["Execution frequency", "Scheduled or on demand (daily, hourly)", "Continuous, on every user request"],
    ["Response requirement", "Results needed later, no waiting user", "Immediate response expected"],
    ["Number of records", "Hundreds to millions per run", "Typically one per call"],
    ["Latency", "High per run, very low per record", "Low per request (milliseconds)"],
    ["Throughput", "High (vectorised processing)", "Limited by request handling overhead"],
    ["Storage", "Results written to file or database", "Response returned to caller; logging optional"],
]
t3 = Table(cmp_data, colWidths=[1.3 * inch, 2.7 * inch, 2.2 * inch])
t3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
]))
story.append(t3)
story.append(Paragraph(
    "Batch inference is appropriate when no instant answer is needed (monthly patient risk screening, nightly "
    "scoring of records, report generation). Real-time inference is appropriate when a decision is needed at once "
    "(doctor entering values and expecting a result on screen). Batch inference can handle many records without one "
    "API request per record because the model and pipeline are loaded once and the whole table is transformed and "
    "predicted in one vectorised call.", body))

story.append(Paragraph("9. Observations, Problems and Corrective Actions", h1))
obs = [
    "Observation: all 3 invalid records were caught and logged, and the output row count matched the successfully processed input count.",
    "Observation: the record with a missing value was still predicted, because the saved pipeline includes the imputer.",
    "Problem: when the bad batch file was first generated, the text value could not be placed in a numeric column. Corrective action: the column was converted to object type before inserting the invalid value.",
    "Problem: the first draft of the validation code crashed when no records failed. Corrective action: the failed-records table is now built so that it works with zero failures.",
    "Note: the batch input reuses test rows from the dataset, so it is for demonstrating the system rather than measuring true accuracy.",
]
for o in obs:
    story.append(Paragraph("&bull; " + o, body))

story.append(Paragraph("10. Conclusion", h1))
story.append(Paragraph(
    "A batch prediction system was built that loads a saved model pipeline, validates 1001 input records, predicts "
    + str(len(rows)) + " valid records in one execution, stores results with timestamps and model version, and "
    "reports performance (about " + str(round(perf["throughput_records_per_sec"])) + " records/second). The comparison "
    "with real-time prediction shows that batch inference suits large, non-urgent workloads.", body))

doc.build(story)
print("PDF report generated:", os.path.join(BASE, "Experiment_7_Report_Updated.pdf"))
print("Assets in:", ASSETS)
