# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

from sklearn.cross_decomposition import PLSRegression

from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Load the data from the Excel file
excel_file_path = 'CORRELATIONS.xlsx'
sheet_name = 'ML'  # Change this to your sheet name
df = pd.read_excel(excel_file_path, sheet_name=sheet_name, engine='openpyxl')

# Define the input columns and labels
input_columns = ['YEAR','BB%','K%','Barrel%','Hard Hit%','EV','LA','LD%','GB%','FB%','Spd','O-Swing%','Whiff%','CSW%','CStr%','SwStr%']

label_column = 'wOBA'

# Split the data into input and labels
x = df[input_columns]
x = x.groupby('YEAR').transform(lambda x: (x - x.mean()) / x.std())
#x = x.to_numpy()
y = df[label_column]
#y = y.to_numpy()

# Split the data into training and testing sets

# Pipeline using PolynomialFeatures and Ridge Regression for deployment in GridSearchCV
pipeline = Pipeline([
    ('poly', PolynomialFeatures(include_bias=True)),
    ('ridge', Ridge())
])

# Define the parameter grid for the polynomial degree and alpha (regularization strength)
param_grid = {
    'poly__degree': [1,2,3,4],  # Specify the degrees to try
    'poly__interaction_only': [True,False],
    'ridge__alpha': [.001,.01, .1, .25, 1.0, 2.5, 5,10.0,20,25]  # Specify the values of alpha to try
}

#Default scoring was an accuracy measurement we want to use MSE
scoring = make_scorer(mean_squared_error, greater_is_better=False)

# GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid,scoring=scoring, cv=5)
# GridSearchCV fit runs the gridsearch algorithm
grid_search.fit(x, y)

cv_scores = grid_search.cv_results_['mean_test_score']
squared_bias = np.mean(cv_scores)
variance = np.std(cv_scores)

# Bias (training error) is given by average_mse
# Variance is given by mse_std

print(f'Average MSE (Bias): {squared_bias}')
print(f'Standard Deviation of MSE (Variance): {variance}')

#Debug purposes writing results from folds for each combo to csv
cvresults= pd.DataFrame(grid_search.cv_results_)
cvresults.to_csv('cv_results.csv')

# Best hyperparamters found by the grid search
print("best params: ", grid_search.best_estimator_.get_params())

# Best estimator which is a pipeline type from the grid search object
#Followed by the specific steps of the pipeline
best_estimator = grid_search.best_estimator_
poly = best_estimator.named_steps['poly']
ridge_model = best_estimator.named_steps['ridge']
print(ridge_model.coef_)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Fit the PolynomialFeatures transformer on x_train
poly.fit(x_train)

# Transform both x_train and x_test
x_train_poly = poly.transform(x_train)
x_test_poly = poly.transform(x_test)

# Fit the ridge model on the transformed training data
ridge_model.fit(x_train_poly, y_train)

# Make predictions on the transformed test data
y_pred = ridge_model.predict(x_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# print("RANDOM FOREST NOW")
# rf = RandomForestRegressor()
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'n_jobs': [-1]
# }

# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(x,y)

# best_rf = grid_search.best_estimator_
# print(best_rf)
# best_rf.fit(x_train, y_train)
# y_pred = best_rf.predict(x_test)
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R-squared: {r2}")

print("ELASTIC NET NOW")
elastic_net = ElasticNet()
param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Adjust values as needed
    'l1_ratio': [0.1, 0.5, 0.9]  # Adjust values as needed
}

grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x, y)
best_elastic_net = grid_search.best_estimator_
best_params = grid_search.best_params_
best_elastic_net.fit(x_train,y_train)
y_pred = best_elastic_net.predict(x_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# PLS Regression with Grid Search CV
print("PLS NOW")
pls = PLSRegression()
param_grid_pls = {'n_components': [1, 2, 3, 4, 5]}
grid_search_pls = GridSearchCV(pls, param_grid_pls, cv=5, scoring='neg_mean_squared_error')
grid_search_pls.fit(x, y)
best_pls = grid_search_pls.best_estimator_
best_pls.fit(x_train,y_train)
y_pred_pls = best_pls.predict(x_test)
mse_pls = mean_squared_error(y_test, y_pred_pls)
r2_pls = r2_score(y_test, y_pred_pls)

print("PLS Regression Results:")
print("Best PLS Components:", best_pls.n_components)
print(f"Mean Squared Error (PLS): {mse_pls}")
print(f"R-squared (PLS): {r2_pls}\n")

# Linear Regression with Grid Search CV
print("LINEAR REGRESSION")
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
# Assuming you have a fitted OLS model (model) and have residuals
residuals = y_test - y_pred_lr

# Breusch-Pagan test
lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, x_test)
print("Breusch-Pagan test p-value:", lm_pvalue)


print("Linear Regression Results:")
print(f"Mean Squared Error (Linear Regression): {mse_lr}")
print(f"R-squared (Linear Regression): {r2_lr}")

# Bayesian Ridge Regression with Grid Search CV
br = BayesianRidge()
param_grid_br = {
    'n_iter': [100, 300, 500],
    'alpha_1': [1e-6, 1e-5, 1e-4],
    'alpha_2': [1e-6, 1e-5, 1e-4],
    'lambda_1': [1e-6, 1e-5, 1e-4],
    'lambda_2': [1e-6, 1e-5, 1e-4]
}
grid_search_br = GridSearchCV(br, param_grid_br, cv=5, scoring='neg_mean_squared_error')
grid_search_br.fit(x, y)
best_br = grid_search_br.best_estimator_
best_br.fit(x_train,y_train)
y_pred_br = best_br.predict(x_test)
mse_br = mean_squared_error(y_test, y_pred_br)
r2_br = r2_score(y_test, y_pred_br)

print("Bayesian Ridge Regression Results:")
print("Best n_iter:", best_br.n_iter)
print("Best alpha_1:", best_br.alpha_1)
print("Best alpha_2:", best_br.alpha_2)
print("Best lambda_1:", best_br.lambda_1)
print("Best lambda_2:", best_br.lambda_2)
print(f"Mean Squared Error (Bayesian Ridge): {mse_br}")
print(f"R-squared (Bayesian Ridge): {r2_br}\n")


# Neural Network (Multi-layer Perceptron) with Grid Search CV
nn = MLPRegressor()
param_grid_nn = {
    'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 25)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}
grid_search_nn = GridSearchCV(nn, param_grid_nn, cv=5, scoring='neg_mean_squared_error')
grid_search_nn.fit(x, y)
best_nn = grid_search_nn.best_estimator_
best_nn.fit(x_train,y_train)
y_pred_nn = best_nn.predict(x_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print("Neural Network (MLP) Results:")
print("Best Hidden Layer Sizes:", best_nn.hidden_layer_sizes)
print("Best Activation Function:", best_nn.activation)
print("Best Alpha (Regularization Strength):", best_nn.alpha)
print(f"Mean Squared Error (Neural Network): {mse_nn}")
print(f"R-squared (Neural Network): {r2_nn}\n")
