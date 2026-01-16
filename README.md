# Multi-Horizon Credit Risk Assessment with Explainability and Fairness Monitoring

This repository contains an end-to-end machine learning system for **credit risk assessment** using the HELOC dataset.  
The system predicts **loan default risk at multiple time horizons**, provides **counterfactual recourse explanations**, ensures **model interpretability using SHAP**, and integrates a **fairness-aware Human-in-the-Loop (HITL) monitoring framework**.

The entire pipeline is implemented in **Python** and designed to run in **Google Colab**.

---

##  Project Overview

The objective of this project is to support **loan approval decision-making** by combining:
- Accurate risk prediction
- Transparent model explanations
- Actionable recourse suggestions
- Fairness monitoring with human oversight

---

##  Key Features

- Multi-horizon credit risk prediction (Current, 6-month, 12-month)
- XGBoost-based classification model
- Robust preprocessing of HELOC-specific special values
- Actionable counterfactual explanations for rejected applicants
- SHAP-based explainable AI (XAI)
- Fairness monitoring using Demographic Parity Difference (DPD)
- Human-in-the-Loop (HITL) escalation for fairness violations
- Interactive custom applicant testing

---

##  Dataset

- **Dataset**: HELOC (Home Equity Line of Credit)
- **Format**: CSV (uploaded as ZIP file)
- **Target Variable**: `RiskPerformance`
  - `Good` → Non-default
  - `Bad` → Default

 The dataset is **not included** in this repository due to privacy and licensing restrictions.

---

##  Workflow

1. Install required Python packages
2. Upload HELOC dataset ZIP file in Google Colab
3. Data preprocessing and feature engineering
4. Train–test split with class stratification
5. Train multi-horizon XGBoost risk models
6. Model evaluation using AUC and classification metrics
7. Generate counterfactual explanations
8. Explain predictions using SHAP
9. Monitor fairness and trigger HITL review when needed
10. Test custom applicants interactively

---

##  Model Description

- **Base Model**: XGBoost Classifier
- **Feature Scaling**: StandardScaler
- **Risk Horizons**:
  - Current risk
  - 6-month projected risk
  - 12-month projected risk

Temporal features (`MSince*`) are adjusted to simulate future risk scenarios.

---

##  Evaluation Metrics

- ROC–AUC
- Precision, Recall, F1-score
- Confusion Matrix
- SHAP feature attributions

---

##  Counterfactual Explanations

For rejected applicants, the system generates **three recourse paths**:
- Quick Fix (high impact, short term)
- Balanced (moderate impact)
- Gradual (low cost, long term)

Each path reports:
- Updated risk score
- Estimated cost
- Timeline for improvement
- Key feature changes

---

##  Fairness & Human-in-the-Loop (HITL)

- **Fairness Metric**: Demographic Parity Difference (DPD)
- **Threshold**: DPD > 0.10 triggers human review
- Fairness checks applied to:
  - Loan approval decisions
  - Recourse recommendations

Model predictions remain unchanged, but flagged cases require **human expert intervention**.

---

##  How to Run (Google Colab)

1. Open the notebook in Google Colab
2. Run all cells sequentially
3. Upload the HELOC dataset ZIP when prompted
4. Review risk predictions, explanations, and fairness reports

---

##  Custom Applicant Testing

The notebook provides functions to:
- Test manually entered applicant data
- Generate multi-horizon risk predictions
- Produce counterfactual recommendations
- Perform fairness-aware HITL checks



For academic and research use only.
