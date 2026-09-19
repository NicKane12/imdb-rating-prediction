#!/usr/bin/env python
# coding: utf-8
"""
Predicting IMDb Movie Ratings
Math 748 Final Project — Nicholas Kane

Combines IMDb's title.basics and title.ratings datasets to study which
film qualities (runtime, release year, genre) predict average rating.
Compares linear regression, ridge/lasso, SVR, and random forests.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor

pd.set_option("display.max_columns", None)
np.random.seed(748)

# ---------------- LOAD DATA ----------------
basics = pd.read_csv(
    "data/title.basics.tsv.gz",
    sep='\t',
    compression='gzip',
    na_values="\\N",
    low_memory=False
)
ratings = pd.read_csv(
    "data/title.ratings.tsv.gz",
    sep="\t",
    compression="gzip",
    na_values="\\N"
)

# ---------------- FILTER & MERGE ----------------
basic_info = basics[basics["titleType"] == "movie"].copy()
basic_info = basic_info[basic_info["isAdult"] == 0].copy()
basic_info = basic_info[["tconst", "primaryTitle", "startYear", "runtimeMinutes", "genres"]]
films = basic_info.merge(ratings, on="tconst", how="inner")

# ---------------- MISSINGNESS ----------------
films["runtime_missing"] = films["runtimeMinutes"].isna()
films["genre_missing"] = films["genres"].isna()

films_modern = films[films["startYear"] >= 1970]
films_clean = films_modern.dropna(subset=["runtimeMinutes", "genres"])

# ---------------- GENRE ENCODING ----------------
genres_coded = films_clean['genres'].str.get_dummies(',')
genre_means = genres_coded.mean()
keep_genres = genre_means[genre_means >= 0.01].index
genres_coded_filtered = genres_coded[keep_genres]

films_model = pd.concat([films_clean, genres_coded_filtered], axis=1)
films_model = films_model.drop(columns=["genres", "runtime_missing", "genre_missing"])

# ---------------- CLEAN RUNTIME ----------------
films_model["runtimeMinutes"] = pd.to_numeric(films_model["runtimeMinutes"], errors="coerce")
films_model = films_model[
    (films_model["runtimeMinutes"] >= 60) &
    (films_model["runtimeMinutes"] <= 300)
]

# ---------------- TRANSFORM VOTES ----------------
films_model["logVotes"] = np.log(films_model["numVotes"])

# ---------------- LINEAR REGRESSION ----------------
X_linear = films_model.drop(
    columns=["tconst", "averageRating", "primaryTitle", "logVotes", "numVotes", "Drama"]
)
y_linear = films_model["averageRating"]

Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=748
)

model = LinearRegression()
model.fit(Xl_train, yl_train)
y_pred = model.predict(Xl_test)
rmse = np.sqrt(mean_squared_error(yl_test, y_pred))
print("Linear RMSE:", rmse)

coef_lin = pd.Series(model.coef_, index=X_linear.columns)
coef_lin[coef_lin > 0].sort_values(ascending=False).head(10).plot(kind="barh", color="teal")
plt.title("LR Coefficients (Positive)")
plt.gca().invert_yaxis()
plt.show()

coef_lin[coef_lin < 0].sort_values().head(10).plot(kind="barh", color="darkorange")
plt.title("LR Coefficients (Negative)")
plt.show()

# ---------------- RIDGE / LASSO ----------------
scaler = StandardScaler()
Xl_train_scaled = scaler.fit_transform(Xl_train)
Xl_test_scaled = scaler.transform(Xl_test)

alphas = np.logspace(-3, 3, 50)

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(Xl_train_scaled, yl_train)
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(Xl_train_scaled, yl_train)
y_pred_ridge = ridge.predict(Xl_test_scaled)
print("Ridge RMSE:", np.sqrt(mean_squared_error(yl_test, y_pred_ridge)))

lasso_cv = LassoCV(alphas=alphas, cv=5)
lasso_cv.fit(Xl_train_scaled, yl_train)

mse_mean = lasso_cv.mse_path_.mean(axis=1)
mse_std = lasso_cv.mse_path_.std(axis=1)
min_idx = np.argmin(mse_mean)
min_error = mse_mean[min_idx]
threshold = min_error + mse_std[min_idx]
alpha_1se = lasso_cv.alphas_[mse_mean <= threshold].max()

lasso_min = Lasso(alpha=lasso_cv.alpha_)
lasso_1se = Lasso(alpha=alpha_1se)
lasso_min.fit(Xl_train_scaled, yl_train)
lasso_1se.fit(Xl_train_scaled, yl_train)

y_pred_min = lasso_min.predict(Xl_test_scaled)
y_pred_1se = lasso_1se.predict(Xl_test_scaled)
print("Lasso RMSE:", np.sqrt(mean_squared_error(yl_test, y_pred_min)))
print("Lasso 1SE RMSE:", np.sqrt(mean_squared_error(yl_test, y_pred_1se)))

# ---------------- SVR ----------------
X_full = films_model.drop(
    columns=["tconst", "averageRating", "primaryTitle", "logVotes", "numVotes"]
)
y_full = films_model["averageRating"]

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=748
)

scaler = StandardScaler()
Xf_train_scaled = scaler.fit_transform(Xf_train)
Xf_test_scaled = scaler.transform(Xf_test)

svr_linear = LinearSVR(max_iter=10000)
svr_linear.fit(Xf_train_scaled, yf_train)
print("Linear SVR RMSE:",
      np.sqrt(mean_squared_error(yf_test, svr_linear.predict(Xf_test_scaled))))

# RBF (subset for speed)
idx = np.random.choice(Xf_train_scaled.shape[0], size=10000, replace=False)
X_small = Xf_train_scaled[idx]
y_small = yf_train.iloc[idx]

svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(X_small, y_small)
print("RBF SVR RMSE:",
      np.sqrt(mean_squared_error(yf_test, svr_rbf.predict(Xf_test_scaled))))

param_grid = {
    "C": [0.1, 1, 10],
    "gamma": [0.01, 0.1, 1],
    "epsilon": [0.05, 0.1, 0.2]
}
grid = GridSearchCV(SVR(kernel="rbf"), param_grid, cv=3,
                     scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X_small, y_small)
best_svr = grid.best_estimator_
print("Tuned SVR RMSE:",
      np.sqrt(mean_squared_error(yf_test, best_svr.predict(Xf_test_scaled))))

# ---------------- RANDOM FOREST ----------------
rf = RandomForestRegressor(n_estimators=200, random_state=748, n_jobs=-1)
rf.fit(Xf_train, yf_train)
print("RF RMSE:", np.sqrt(mean_squared_error(yf_test, rf.predict(Xf_test))))

rf_tune = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    random_state=748,
    n_jobs=-1
)
rf_tune.fit(Xf_train, yf_train)
print("Tuned RF RMSE:", np.sqrt(mean_squared_error(yf_test, rf_tune.predict(Xf_test))))
