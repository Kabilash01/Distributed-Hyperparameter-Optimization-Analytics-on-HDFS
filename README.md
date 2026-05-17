# HPO-B Big Data Analytics Project

## Overview
A distributed hyperparameter optimization analytics system built on
HDFS + Hive + PySpark + MLflow, ingesting 3.27 million real training
evaluations from the HPO-B NeurIPS 2021 benchmark.

## Key Results
| Metric | Value |
|--------|-------|
| Real HPO evaluations | 3,279,050 |
| Search spaces | 16 |
| Datasets | 97 |
| Bayesian TPE best accuracy | 0.9750 |
| Trials to 90% accuracy | 10 vs 24 (2.4x faster) |
| Recommendation hit rate | 99% |
| Surrogate R² test | 0.8652 |
| MLflow runs logged | 8 |

## Tech Stack
- HDFS (Hadoop 3.x pseudo-distributed)
- Apache Hive 4.0 + HiveQL
- PySpark + MLlib
- Optuna (TPE + CMA-ES)
- MLflow experiment tracking
- scikit-learn RandomForest surrogate

## Architecture
HDFS → Hive → PySpark → Bayesian Optimizer → Meta-Learner → MLflow

