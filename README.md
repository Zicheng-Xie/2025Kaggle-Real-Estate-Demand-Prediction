# 2025Kaggle-Real-Estate-Demand-Prediction
A _**Top 13% (95th/777 teams)**_ implementation for 2025 Kaggle Real Estate Demand Prediction Competition

# Real Estate Demand Prediction – Final Solution

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Competition](https://img.shields.io/badge/Kaggle-Real%20Estate%20Demand%202025-brightgreen.svg)
![Model](https://img.shields.io/badge/Model-CatBoost%20%2B%20Time--Series-orange.svg)
![Status](https://img.shields.io/badge/Status-Final%20Submission-success.svg)

## 🎓 Introduction

This repository contains the final solution for the *Kaggle 2025 Real Estate Demand Prediction* task, in which the goal is to forecast monthly transaction amounts of new residential properties at the **(city sector × month)** level. The project formulates the problem as a panel time-series regression task with strong seasonality and severe penalties for large relative errors, closely aligned with the competition’s two-stage evaluation metric.

## 🧠 Methodological Framework

The pipeline constructs a **complete sector–month panel** by generating the Cartesian product of sectors and calendar months and then left-joining all available data sources, including new-house transactions, secondary-market activity, land transactions, points-of-interest (POI) statistics, and yearly city indices. A carefully designed time parser converts heterogeneous month strings into integer time indices and separates year, month, and quarter components, enabling consistent temporal alignment across tables. All merges and aggregations are performed in a leakage-free manner so that each row’s features depend only on information available up to, or at, its corresponding month.

## 📈 Problem Formulation and Targets

The target variable is defined as the **next-month new-house transaction amount** for each sector. After ordering the panel by sector and time, the target is generated via a one-step temporal shift, ensuring that the model always predicts future demand from current and historical information. Missing records in the raw transaction table are interpreted as structural zeros, in accordance with the competition design, and the training set is restricted to months with observed targets. The solution further derives a binary label indicating whether demand is strictly positive, which is subsequently used in a two-stage modeling framework.

## 🕒 Temporal and Seasonal Feature Design

The feature set is dominated by **time-series and seasonal descriptors**. For each sector, the pipeline constructs multi-order lags and rolling statistics (means and medians) for the target and key covariates (e.g., neighboring-sector transactions, pre-sale areas, land-deal quantities). These include short-term lags (1–3 months) to capture local momentum and longer-term lags (6–12 months) to encode annual cycles. In addition, the code introduces harmonic seasonal encodings via sine and cosine transformations with 6- and 12-month periods, as well as explicit indicators for calendar quarter. Domain-specific seasonal structure is incorporated through hand-engineered flags for **Lunar New Year** months and their immediate predecessors, and through multiplicative corrections for December, reflecting empirical patterns of year-end sales surges.

## 🔧 Data Engineering and Robustness

The repository implements a robust data-engineering layer that standardizes data types, removes duplicate sector–month rows, and aggregates conflicting entries using numeric means and categorical first occurrences. Potential infinities are converted to missing values, while numerical features are imputed to a constant and categorical features are assigned a dedicated missing token before modeling. All transformations are organized as reusable functions, allowing consistent processing of training and test panels and minimizing accidental information leakage between time periods.

## 🧮 Seasonal Baselines

Before fitting machine-learning models, the solution constructs **classical time-series baselines** on each sector-specific univariate series of transaction amounts. Two complementary forecasts are computed: a Holt–Winters exponential smoothing model with annual seasonality, and an exponentially weighted geometric mean extrapolation. Both baselines are subsequently adjusted by sector-specific December multipliers and mapped back to the test panel. These forecasts serve both as **competitive standalone predictors** and as **fallback / shrinkage targets** for the learned models, significantly improving robustness against extreme prediction errors under the competition’s stringent scoring regime.

## 🤖 Two-Stage CatBoost Modeling

The predictive core is a **two-stage CatBoost architecture**. In the first stage, a CatBoost classifier models the probability that next-month demand is strictly positive, using the full feature set derived from the panel. In the second stage, a CatBoost regressor models the logarithm of the positive demand (`log1p(y)`), again using the same covariates. The regression model is trained with an absolute-error-oriented loss to better match the competition’s scale-invariant error measure. The final demand prediction combines the regressor output (after exponentiation) with the seasonal baseline via a convex ensemble, followed by sector-wise caps based on historical high quantiles and a probability gate derived from the classifier; this design explicitly aims to reduce the fraction of samples with very large relative error, which are heavily penalized by the evaluation metric.

## 🧪 Evaluation and Cross-Validation

Model performance is assessed through **time-series cross-validation** using `TimeSeriesSplit`, ensuring that each validation fold corresponds to a future time interval relative to its training data. This setup mirrors the real deployment scenario of forecasting upcoming months from past observations. Out-of-fold predictions from all splits are averaged on the test set, yielding a temporally informed ensemble that stabilizes predictions across different historical windows. The final notebook produces a competition-ready submission file with correctly formatted identifiers and predicted new-house transaction amounts for all required sector–month pairs.

