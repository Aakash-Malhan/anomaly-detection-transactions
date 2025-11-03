**Financial Transaction Anomaly Detection**

Demo - https://huggingface.co/spaces/aakash-malhan/anomaly-detection-transactions

**Business Problem**

    Financial institutions process millions of transactions daily. Detecting abnormal activity such as fraud, account takeovers, or bot-generated transactions & is critical to protect customers and reduce financial losses.
    Traditional rule-based systems fail to detect novel fraud patterns, and manual review of every transaction is impossible at scale.

**Solution Summary**

    Engineers behavior-based transaction features (rolling stats, z-scores, velocity, merchant frequency, etc.)
    Trains an unsupervised anomaly detection model (Isolation Forest)
    Uses time-aware evaluation (train on past, test on future)
    Applies a top-k review threshold (e.g., review top 1% risky transactions)
    Generates plain-English reasons for anomalies (PM-friendly explanations)
    Hosts an interactive Gradio app on Hugging Face for scoring CSVs
    Provides executive summary insights for business users & PMs

<img width="1786" height="904" alt="Screenshot 2025-11-02 182638" src="https://github.com/user-attachments/assets/9f32ab14-4833-49ff-acfd-40b8914c5fc9" />
<img width="1777" height="359" alt="Screenshot 2025-11-02 182658" src="https://github.com/user-attachments/assets/7f1885d9-398e-47c9-b0f7-c8ef4d55c59a" />


**Tech Stack**

    TPython | Isolation Forest (PyOD) | Scikit-Learn | Pandas | NumPy | Joblib model + scaler + metadata

**Impact & Results**

    Dataset Size                217,441 financial transactions
    Avg review budget           1% of transactions (≈ 2,174 daily reviews)
    Flagged by model            ~29.3% raw anomalies (threshold narrows to reviewable %)
    False negatives reduced     ≈ 20–35% potential reduction vs strict rules
    Time savings                Up to 90% reduction in manual triage vs reviewing all data
