# Treatment-Aware Telecom Churn Targeting

An end-to-end data science project for deciding **which customers should receive
a retention intervention**. It uses a real randomized telecom campaign dataset
and evaluates incremental impact rather than treating churn prediction as a
complete business solution.

## Why this project is different

A conventional churn model answers: “Who is likely to leave?”

A retention team needs a harder answer: “Whose churn probability will decrease
because we contact them?” High-risk customers can be:

- persuadable;
- sure things who would stay anyway;
- lost causes who will leave regardless;
- customers for whom an intervention is counterproductive.

This repository compares risk targeting with S- and T-learner uplift models,
selects a policy from cross-fitted predictions, and evaluates it once on an
untouched holdout set.

## Dataset

The project uses
[OpenML 45580: churn-uplift-mlg](https://www.openml.org/d/45580), released by
the Machine Learning Group at ULB from randomized Orange Belgium retention
campaigns.

- 11,896 customers
- 160 anonymized numerical PCA components
- 18 anonymized categorical factors
- `t`: randomized treatment assignment
- `y`: churn within two months
- campaign period: September–December 2020
- OpenML upload: July 6, 2023
- peer-reviewed dataset paper: ECML PKDD proceedings, 2025
- license: CC BY-NC-ND

See [data/DATASET.md](data/DATASET.md) for provenance, checksum, citation, and
limitations. The raw file is not committed to Git.

## Methodology

1. Validate the official ARFF schema and checksum.
2. Reserve a stratified 20% holdout before model selection.
3. Generate out-of-fold predictions on the remaining 80%.
4. Compare:
   - CatBoost risk targeting trained on untreated customers;
   - CatBoost S-learner;
   - CatBoost T-learner;
   - logistic-regression T-learner baseline.
5. Select the model by OOF AUUC, with Qini and Uplift@K as supporting metrics.
6. Choose campaign size using estimated incremental value and configurable costs.
7. Refit on all development data.
8. Evaluate once on holdout with bootstrap confidence intervals.

The split is stratified by the joint `(treatment, churn)` class. A temporal split
would be preferable, but the public dataset does not expose campaign timestamps.

## Metrics

- AUUC and Qini for uplift ranking;
- Uplift@K and estimated prevented churns;
- expected campaign net value;
- ROC-AUC, average precision, Brier score, and log loss for factual outcomes;
- bootstrap 95% confidence intervals on holdout.

Accuracy is intentionally not a primary metric because churn is rare.

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

The dependency set includes a lightweight Jupyter kernel for the notebook but
does not install the full JupyterLab server. Open the notebook in VS Code,
PyCharm, Codex, or an existing Jupyter installation.

Download the dataset from OpenML and verify its MD5 checksum:

```bash
python -m src.download_data
```

## Run

Train all candidates and create reports:

```bash
python -m src.train
```

Run tests:

```bash
pytest
```

Score a batch with the saved model:

```bash
python -m src.predict \
  --input data/processed/customers_to_score.csv \
  --output data/processed/scored_customers.csv
```

Recommendations are made as a **top fraction within a batch**, because campaign
budgets are batch decisions and score scales can shift after retraining.
If the selected model is the risk-only benchmark, the treatment counterfactual
column is intentionally empty: the model supplies a ranking score, not an
individual causal-effect claim.

## Configurable business assumptions

```bash
python -m src.train \
  --customer-value 500 \
  --contact-cost 10 \
  --offer-cost 40 \
  --max-contact-rate 0.30
```

These values are assumptions, not facts inferred from the anonymized dataset. If
all candidate policies have negative estimated value, the correct output is “do
not launch this campaign under the current economics.”

## Outputs

- `models/best_model.joblib` — fitted model, schema, policy, and metadata;
- `reports/final_report.md` — Russian-language analytical report;
- `reports/metrics/model_comparison.csv` — OOF model comparison;
- `reports/metrics/results.json` — reproducibility metadata and holdout metrics;
- `reports/metrics/*_predictions.csv` — auditable OOF/holdout predictions;
- `reports/figures/` — uplift, calibration, value, and importance plots.

## Project structure

```text
.
├── data/
│   ├── DATASET.md
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_eda_uplift_modeling.ipynb
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── final_report.md
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── download_data.py
│   ├── evaluation.py
│   ├── models.py
│   ├── predict.py
│   ├── reporting.py
│   ├── train.py
│   ├── uplift_metrics.py
│   └── utils.py
└── tests/
```

## Interpretation limits

Feature importance refers to anonymized components and must not be described as
human-readable churn causes. Predictive importance is not causal importance.
Business effectiveness must ultimately be confirmed in a new randomized
campaign.
