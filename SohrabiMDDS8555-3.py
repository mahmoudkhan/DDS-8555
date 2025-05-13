# -*- coding: utf-8 -*-
"""
Created on Mon May 12 00:44:13 2025

@author: msohr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from ISLP.models import ModelSpec as MS
import sklearn.model_selection as skm
import sklearn.linear_model as skl

# Set random seed for reproducibility
np.random.seed(55)

# --- 1. Data Loading and Basic Preprocessing ---

# Load the Abalone dataset
df = pd.read_csv('c:/Users/msohr/Desktop/NU/DDS-8555/Abalone/train.csv')

from sklearn.preprocessing import StandardScaler

# One-hot encoding for 'Sex'
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Separate features and target
design = MS(df.columns.drop(['Rings', 'id'])).fit(df)
D = design.fit_transform(df)
D = D.drop('intercept', axis=1)
X = np.asarray(D)
#X = design.transform(df)
Y = df['Rings']

K = 5
kfold = skm.KFold(K, random_state=55, shuffle=True)
scaler = StandardScaler(with_mean=True, with_std=True)
lassoCV = skl.ElasticNetCV(n_alphas=100, l1_ratio=1, cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV)])
pipeCV.fit(X, Y)
tuned_lasso = pipeCV.named_steps['lasso']
print("tuned_lasso.alpha_",tuned_lasso.alpha_)

# Coefficients and selected features
lasso_coef = pd.Series(tuned_lasso.coef_, index=D.columns)
selected_features = lasso_coef[lasso_coef != 0]
print("Selected features:\n", selected_features)

y_pred = tuned_lasso.predict(X)
y_pred = np.maximum(0, y_pred)
residuals = Y - y_pred

# Residual plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Rings')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
print(f'Durbin-Watson statistic: {dw:.2f}')

import scipy.stats as stats

sns.histplot(residuals, kde=True)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Check VIFs
vif_data = pd.DataFrame()
vif_data["feature"] = D.columns
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print(vif_data)

# --- Load test data ---
test_df = pd.read_csv("C:/Users/msohr/Desktop/NU/DDS-8555/Abalone/test.csv")
test_ids = test_df["id"]
test_df = pd.get_dummies(test_df, columns=['Sex'], drop_first=True)
design2 = MS(test_df.columns.drop('id')).fit(test_df)
D2 = design2.fit_transform(test_df)
D2 = D2.drop('intercept', axis=1)
X_test = np.asarray(D2)
# --- Predict using Model ---
preds_tuned_lasso = tuned_lasso.predict(X_test)
preds_tuned_lasso = np.maximum(0, preds_tuned_lasso)

# --- Save submission files ---
submission = pd.DataFrame({'id': test_ids, 'Rings': preds_tuned_lasso})
submission.to_csv("submission_model.csv", index=False)
print("Submissions saved: submission_model.csv")

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

# Apply PCA
pca = PCA(random_state=55)
Xs = X - X.mean(0)[None, :]
X_scale = X.std(0)
Xs = Xs / X_scale[None, :]
X_pca = pca.fit_transform(Xs)
# Choose number of components (e.g., 95% variance)
explained_var = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(explained_var >= 0.95) + 1
print(f"Number of components to explain 95% variance: {n_components}")

pca = PCA(n_components=n_components, random_state=55)
linreg = skl.LinearRegression()
pipe = Pipeline([('scaler', scaler), ('pca', pca), ('linreg', linreg)])
pipe.fit(Xs, Y)
pcr_model = pipe.named_steps['linreg']
print(pcr_model.coef_)

# Regression Coefficients for PCs
pcr_coeffs = pipe.named_steps['linear_regression'].coef_
print("PCR Coefficients (for each Principal Component predicting Rings):")
for i, coeff in enumerate(pcr_coeffs):
    print(f"PC{i+1}: {coeff:.4f}")
print("Interpretation: A coefficient for PCk tells how log(1+Rings) changes for a one-unit increase in PCk.")

# Predictions
y_pred = pipe.predict(Xs)
y_pred = np.maximum(0, y_pred)

residuals = Y - y_pred

# Residual plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Rings')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

dw = durbin_watson(residuals)
print(f'Durbin-Watson statistic: {dw:.2f}')

sns.histplot(residuals, kde=True)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.show()

# Check VIFs
vif_data = pd.DataFrame()
vif_data["feature"] = D.columns
vif_data["VIF"] = [variance_inflation_factor(Xs, i) for i in range(Xs.shape[1])]
print(vif_data)

# --- Load test data ---
test_df = pd.read_csv("C:/Users/msohr/Desktop/NU/DDS-8555/Abalone/test.csv")
test_ids = test_df["id"]
test_df = pd.get_dummies(test_df, columns=['Sex'], drop_first=True)
design2 = MS(test_df.columns.drop('id')).fit(test_df)
D2 = design2.fit_transform(test_df)
D2 = D2.drop('intercept', axis=1)
X_test = np.asarray(D2)
# --- Predict using Model ---
preds_pcr = pipe.predict(X_test)
preds_pcr = np.maximum(0, preds_pcr)

# --- Save submission files ---
submission = pd.DataFrame({'id': test_ids, 'Rings': preds_pcr})
submission.to_csv("submission_model.csv", index=False)
print("Submissions saved: submission_model.csv")
