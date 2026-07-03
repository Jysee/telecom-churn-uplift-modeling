# Dataset: Orange Belgium churn uplift

The project uses **OpenML dataset 45580, `churn-uplift-mlg`**.

- Source: Orange Belgium retention campaigns and the Machine Learning Group at ULB
- Campaign period: September–December 2020
- Published as an OpenML dataset: 6 July 2023
- Peer-reviewed publication: ECML PKDD proceedings, 2025
- Samples: 11,896
- Predictors: 160 anonymized PCA components and 18 anonymized categorical factors
- Outcome: `y` — churn within two months after the campaign
- Treatment: `t` — assignment to the retention intervention
- License: CC BY-NC-ND
- OpenML: https://www.openml.org/d/45580
- Paper: https://doi.org/10.1007/978-3-031-74640-6_21
- Benchmark code: https://github.com/TheoVerhelst/Churn-Uplift-Dataset-Paper

The raw file is intentionally excluded from Git. Download and validate it with:

```bash
python -m src.download_data
```

Expected MD5:

```text
bd328613c1c4ef793ab04cd4dd1c6db0
```

## Why this dataset

Unlike the common IBM Telco demo dataset, this is a real randomized retention
campaign dataset. It supports treatment-aware evaluation: the project can
estimate who is likely to benefit from an intervention, not merely who is likely
to churn.

## Important limitation

The predictors are anonymized. Numerical variables are PCA components and
categorical levels have generic names. Therefore feature importance describes
predictive signal, not human-readable business causes. The project deliberately
does not invent semantic interpretations for anonymized features.
