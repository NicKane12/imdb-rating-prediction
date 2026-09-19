# Predicting IMDb Movie Ratings

## Overview

This project uses IMDb's public datasets to study which qualities of a feature film influence its average user rating. Linear regression, ridge/lasso, support vector regression, and random forests are compared to identify the strongest predictors and find an adequate model for prediction.

## Methods

- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression (linear and RBF kernel, untuned and tuned)
- Random Forest (untuned and tuned)

## Evaluation

Models were evaluated using RMSE (root mean squared error) on a held-out 20% test set, with average rating (1–10 scale) as the response variable.

## Key Findings

- Tuned Random Forest performed best overall, with an **RMSE of 1.20** (vs. 1.23 for baseline linear regression).
- **Documentary**, **Horror**, and **runtime** were consistently the strongest predictors of average rating across nearly every model — documentaries trend higher, horror trends lower.
- Regularization (ridge/lasso) and untuned SVR gave only marginal RMSE improvements over plain linear regression, while sacrificing interpretability.
- Release year mattered more than expected in the random forest models, likely reflecting IMDb's user base skewing toward certain eras.

## Data

Combined from two IMDb public datasets: `title.basics` (movie metadata) and `title.ratings` (rating scores and vote counts), filtered to feature-length films released from 1970 onward. Source: [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/).

## Repo Structure

- `analysis/`: Python script containing data cleaning, EDA, and modeling.
- `report/`: final project report and slides.
- `data/`: raw data used in the project (not included due to size — see Data section for source).

## How to Run

1. Download `title.basics.tsv.gz` and `title.ratings.tsv.gz` from the IMDb datasets page into `data/`.
2. Open `analysis/imdb_rating_prediction.py`.
3. Install required packages listed at the top of the file (`pandas`, `numpy`, `matplotlib`, `scikit-learn`).
4. Run the script sequentially to reproduce the cleaning, EDA, and modeling steps.

## AI Use

ChatGPT was used to help with Python scripts and syntax. All written analysis and interpretation are the author's own.
