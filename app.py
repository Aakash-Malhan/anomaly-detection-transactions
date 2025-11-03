import json, os, io, joblib, pandas as pd, numpy as np, gradio as gr

# Load artifacts
SCALER = joblib.load("artifacts/scaler.pkl")
MODEL  = joblib.load("artifacts/model.pkl")
META   = json.load(open("artifacts/metadata.json"))

FEATURES = META["features"]             
THRESH  = META.get("threshold", None)
BEST    = META.get("best_model", "iforest")

# Path to the default sample CSV in your repo 
DEFAULT_CSV = "sample_data/financial_anomaly_data.csv"
TS_COL = "Timestamp"  # original timestamp column name from your raw file

# Feature engineering
def _freq_encode(series: pd.Series) -> pd.Series:
    freq = series.value_counts(dropna=False)
    return series.map(freq).astype(float)

def build_features_from_raw(df_raw: pd.DataFrame, ts_col: str = TS_COL) -> pd.DataFrame:
    df = df_raw.copy()

    # 1) Parse and sort by time
    if ts_col not in df.columns:
        raise ValueError(f"Missing required timestamp column '{ts_col}' in uploaded CSV.")
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(ts_col).reset_index(drop=True)

    # 2) Basic time features
    df["hour"]  = df[ts_col].dt.hour
    df["dow"]   = df[ts_col].dt.dayofweek
    df["dom"]   = df[ts_col].dt.day
    df["month"] = df[ts_col].dt.month

    # 3) Amount transforms
    if "Amount" not in df.columns:
        raise ValueError("Missing required column 'Amount' in uploaded CSV.")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Amount"] = df["Amount"].fillna(df["Amount"].median())
    df["amt_log"] = np.log1p(df["Amount"])

    # 4) Frequency encodings for common categoricals (create if absent)
    for col in ["Merchant", "TransactionType", "Location", "AccountID"]:
        if col in df.columns:
            df[f"{col}_freq"] = _freq_encode(df[col].astype("category"))
        else:
            df[f"{col}_freq"] = 0.0

    # 5) Rolling behavior per AccountID (or global fallback)
    if "AccountID" in df.columns:
        grp = df.groupby("AccountID", sort=False)
        df["acc_roll_mean"]   = grp["Amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        df["acc_roll_std"]    = grp["Amount"].transform(lambda s: s.rolling(20, min_periods=5).std())
        df["acc_prev_amt"]    = grp["Amount"].shift(1)
        df["acc_txn_ct_w50"]  = grp["Amount"].transform(lambda s: s.rolling(50, min_periods=5).count())
        df["acc_amt_sum_w50"] = grp["Amount"].transform(lambda s: s.rolling(50, min_periods=5).sum())
    else:
        df["acc_roll_mean"]   = df["Amount"].rolling(20, min_periods=5).mean()
        df["acc_roll_std"]    = df["Amount"].rolling(20, min_periods=5).std()
        df["acc_prev_amt"]    = df["Amount"].shift(1)
        df["acc_txn_ct_w50"]  = df["Amount"].rolling(50, min_periods=5).count()
        df["acc_amt_sum_w50"] = df["Amount"].rolling(50, min_periods=5).sum()

    df["acc_roll_std"]  = df["acc_roll_std"].replace([np.inf, -np.inf], np.nan)
    df["acc_roll_mean"] = df["acc_roll_mean"].fillna(df["acc_roll_mean"].median())
    df["acc_roll_std"]  = df["acc_roll_std"].fillna(df["acc_roll_std"].median())
    df["acc_prev_amt"]  = df["acc_prev_amt"].fillna(df["acc_prev_amt"].median())

    df["acc_z"]         = (df["Amount"] - df["acc_roll_mean"]) / (df["acc_roll_std"] + 1e-6)
    df["amt_diff_prev"] = df["Amount"] - df["acc_prev_amt"]

    # numeric only; drop timestamp from feature matrix
    num_df = df.select_dtypes(include=[np.number]).copy()
    for c in num_df.columns:
        col = num_df[c].replace([np.inf, -np.inf], np.nan)
        num_df[c] = col.fillna(col.median())

    fe = num_df  # already numeric
    return df[[ts_col]].join(fe)

# Light rule-based explanations
def make_reason_rules(feature_df: pd.DataFrame) -> dict:
    """Precompute dataset-based thresholds for human explanations."""
    # Percentile cutoffs (robust to scale)
    q = feature_df.quantile
    rules = {
        "big_amt":             q(0.99).get("Amount", np.nan),
        "big_jump":            q(0.95).get("amt_diff_prev", np.nan),
        "burst_activity":      q(0.95).get("acc_txn_ct_w50", np.nan),
        "rare_merchant_freq":  q(0.05).get("Merchant_freq", np.nan),
        "rare_location_freq":  q(0.05).get("Location_freq", np.nan),
        "late_hour_amt":       q(0.75).get("Amount", np.nan),  # high amount in off-hours
        "high_z":              3.0,  # 3σ from account pattern
    }
    return rules

def explain_rows(original_df: pd.DataFrame, engineered_full: pd.DataFrame, scores: np.ndarray) -> tuple[pd.Series, dict]:
    """
    Return reasons per row (string) and an aggregate counter for an executive summary.
    engineered_full includes Timestamp + numeric features.
    """
    F = engineered_full  # includes TS_COL + features
    feature_df = F.drop(columns=[TS_COL], errors="ignore")
    rules = make_reason_rules(feature_df)
    counters = {k: 0 for k in ["High amount", "Unusual jump vs last txn",
                               "Burst of recent activity", "Rare merchant",
                               "Rare location", "Odd hour high amount", "Deviates from account pattern"]}

    reasons = []
    for i in range(len(original_df)):
        r = []

        # 1) High absolute amount
        if "Amount" in feature_df.columns and not np.isnan(rules["big_amt"]):
            if feature_df.iloc[i]["Amount"] >= rules["big_amt"]:
                r.append("High absolute amount")
                counters["High amount"] += 1

        # 2) Large jump since previous
        if "amt_diff_prev" in feature_df.columns and not np.isnan(rules["big_jump"]):
            if feature_df.iloc[i]["amt_diff_prev"] >= rules["big_jump"]:
                r.append("Unusual jump vs last txn")
                counters["Unusual jump vs last txn"] += 1

        # 3) Recent burst of activity
        if "acc_txn_ct_w50" in feature_df.columns and not np.isnan(rules["burst_activity"]):
            if feature_df.iloc[i]["acc_txn_ct_w50"] >= rules["burst_activity"]:
                r.append("Burst of recent activity")
                counters["Burst of recent activity"] += 1

        # 4) Rare merchant/location by frequency
        if "Merchant_freq" in feature_df.columns and not np.isnan(rules["rare_merchant_freq"]):
            if feature_df.iloc[i]["Merchant_freq"] <= rules["rare_merchant_freq"]:
                r.append("Rare merchant for cohort")
                counters["Rare merchant"] += 1
        if "Location_freq" in feature_df.columns and not np.isnan(rules["rare_location_freq"]):
            if feature_df.iloc[i]["Location_freq"] <= rules["rare_location_freq"]:
                r.append("Rare location for cohort")
                counters["Rare location"] += 1

        # 5) Off-hour + high amount
        if "hour" in feature_df.columns and "Amount" in feature_df.columns and not np.isnan(rules["late_hour_amt"]):
            h = int(feature_df.iloc[i]["hour"])
            if (h <= 5 or h >= 23) and feature_df.iloc[i]["Amount"] >= rules["late_hour_amt"]:
                r.append("Odd hour high amount")
                counters["Odd hour high amount"] += 1

        # 6) Deviates from account pattern (z-score)
        if "acc_z" in feature_df.columns:
            if feature_df.iloc[i]["acc_z"] >= rules["high_z"]:
                r.append("Deviates from account pattern (3σ+)")
                counters["Deviates from account pattern"] += 1

        # Always include a fallback with the model score
        if not r:
            r = [f"Model isolation score = {scores[i]:.3f} (outlier in feature space)"]
        reasons.append("; ".join(r))

    return pd.Series(reasons, index=original_df.index, name="reasons"), counters

# Scoring
def _align_and_score(engineered_full: pd.DataFrame, original_df: pd.DataFrame):
    # Ensure all training-time features exist; fill with median if missing
    work = engineered_full.copy()
    if TS_COL in work.columns:
        work_no_ts = work.drop(columns=[TS_COL])
    else:
        work_no_ts = work

    for c in FEATURES:
        if c not in work_no_ts.columns:
            work_no_ts[c] = np.nan
    work_no_ts = work_no_ts[FEATURES].copy()
    for c in work_no_ts.columns:
        col = pd.to_numeric(work_no_ts[c], errors="coerce")
        col = col.replace([np.inf, -np.inf], np.nan)
        work_no_ts[c] = col.fillna(work_no_ts[c].median())

    Xs = SCALER.transform(work_no_ts.values)

    try:
        scores = MODEL.decision_function(Xs)   # PyOD iforest
    except Exception:
        scores = -MODEL.score_samples(Xs)      # sklearn fallback

    flagged = (scores >= THRESH).astype(int) if THRESH is not None else None

    # Explanations (row-wise + aggregate)
    reason_series, counters = explain_rows(original_df, engineered_full, scores)

    out = original_df.copy()
    out["anomaly_score"] = scores
    if flagged is not None:
        out["flag"] = flagged

    # Add reasons to the top anomalies table
    out["reasons"] = reason_series

    # Summary for the JSON box
    if flagged is not None:
        summary = {
            "review_queue": int(out["flag"].sum()),
            "total": int(len(out)),
            "percent_flagged": round(100.0 * out["flag"].mean(), 3),
            "threshold_used": float(THRESH),
            "model": BEST
        }
    else:
        summary = {
            "suggested_threshold_top1pct": float(np.quantile(scores, 0.99)),
            "model": BEST
        }

    # Executive summary (markdown) for PMs/recruiters
    topk = out.sort_values("anomaly_score", ascending=False).head(25).copy()
    top_reasons = topk["reasons"].values
    # Convert counters to percentages on the full dataset and on top-k
    def pct(n, d): return 0.0 if d == 0 else round(100.0 * n / d, 2)
    exec_lines = [
        f"**Model:** {BEST} &nbsp;|&nbsp; **Threshold:** {summary.get('threshold_used', '—')} &nbsp;|&nbsp; "
        f"**Flagged:** {summary.get('review_queue', 0)}/{summary.get('total', len(out))} "
        f"({summary.get('percent_flagged', '—')}%)",
        "",
        "**What drives anomalies (top patterns):**",
    ]
    # Build a quick view of top patterns using the aggregate counters
    pattern_map = [
        ("High amount", "Unusually large amounts relative to cohort"),
        ("Unusual jump vs last txn", "Sudden spike compared to previous transaction"),
        ("Burst of recent activity", "Dense cluster of recent transactions"),
        ("Rare merchant", "Merchant rarely seen in the dataset/account cohort"),
        ("Rare location", "Location rarely seen in the dataset/account cohort"),
        ("Odd hour high amount", "High-value activity during off-hours"),
        ("Deviates from account pattern", "3σ+ deviation from the account’s usual spend"),
    ]
    total_n = len(out)
    exec_lines.append("")
    for key, human in pattern_map:
        exec_lines.append(f"- **{human}** — approx **{pct(counters.get(key,0), total_n)}%** of transactions; higher concentration in the top anomalies.")
    exec_lines.append("")
    exec_lines.append("**Each anomaly’s reason(s):** Visible in the table under the `reasons` column (e.g., “High absolute amount; Rare merchant; Odd hour high amount”).")

    executive_md = "\n".join(exec_lines)

    # Return
    return summary, topk[original_df.columns.tolist() + ["anomaly_score"] + (["flag"] if flagged is not None else []) + ["reasons"]], executive_md

# Entry point when user uploads a CSV
def score_csv(file_obj):
    if file_obj is None:
        return {"error": "Please upload a CSV or click 'Run sample'."}, pd.DataFrame(), "Upload a CSV to see an executive summary."
    # gr.File returns a tempfile path or a dict-like; support both
    path = file_obj.name if hasattr(file_obj, "name") else getattr(file_obj, "tmp_name", None)
    if path is None and isinstance(file_obj, dict) and "name" in file_obj:
        path = file_obj["name"]
    if path is None:
        # fallback: read bytes if provided
        content = file_obj.read() if hasattr(file_obj, "read") else None
        if content is None:
            return {"error": "Unrecognized file object."}, pd.DataFrame(), "Could not read file."
        df_raw = pd.read_csv(io.BytesIO(content))
    else:
        df_raw = pd.read_csv(path)

    # If the CSV already contains engineered columns, use them; otherwise build them from raw
    has_all_features = all(c in df_raw.columns for c in FEATURES)
    if has_all_features:
        engineered_full = df_raw.copy()
    else:
        engineered_full = build_features_from_raw(df_raw, TS_COL)
    return _align_and_score(engineered_full, df_raw)

# One-click sample runner
def run_sample():
    if not os.path.exists(DEFAULT_CSV):
        return {"error": f"Sample not found at {DEFAULT_CSV}. Upload a CSV instead."}, pd.DataFrame(), "Sample not found."
    df_raw = pd.read_csv(DEFAULT_CSV)
    engineered_full = build_features_from_raw(df_raw, TS_COL)
    return _align_and_score(engineered_full, df_raw)

# UI
with gr.Blocks(title="Anomaly Detection in Transactions") as demo:
    gr.Markdown(
        "## Anomaly Detection in Transactions\n"
        "Scores transactions using Isolation Forest and a saved top-k review threshold.\n"
        "**Upload a CSV** with raw columns (`Timestamp, AccountID, Amount, Merchant, TransactionType, Location, ...`) "
        "or with engineered feature columns. Missing columns are handled automatically.\n"
    )

    with gr.Row():
        file_in = gr.File(label="Upload CSV", file_types=[".csv"])
        run_btn = gr.Button("Run sample (default dataset)")

    summary = gr.JSON(label="Summary")
    topk = gr.Dataframe(label="Top 25 Anomalies (includes `reasons`)")
    exec_md = gr.Markdown(label="Executive summary")

    file_in.change(fn=score_csv, inputs=file_in, outputs=[summary, topk, exec_md])
    run_btn.click(fn=run_sample, inputs=None, outputs=[summary, topk, exec_md])

    # Also show an Example (clickable)
    if os.path.exists(DEFAULT_CSV):
        gr.Examples(
            examples=[[DEFAULT_CSV]],
            inputs=[file_in],
            label="Example: default dataset",
            examples_per_page=1
        )

if __name__ == "__main__":
    demo.launch()
