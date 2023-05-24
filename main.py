import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

if __name__ == "__main__":

    # Create a pipeline with PolynomialFeatures and LinearRegression
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(include_bias=True)),
        ('regression', Ridge())
    ])

    # Define the parameter grid for the polynomial degree and alpha (regularization strength)
    param_grid = {
        'poly__degree': [1, 2, 3],  # Specify the degrees to try
        'regression__alpha': [.001,.01, .1, .25, 1.0, 2.5, 10.0]  # Specify the values of alpha to try
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)

    # Fit the GridSearchCV on your data
    grid_search.fit(x, y)

    # Get the best degree and alpha found by the grid search
    best_degree = grid_search.best_params_['poly__degree']
    best_alpha = grid_search.best_params_['regression__alpha']

    # Transform the data using the best degree
    x_poly = PolynomialFeatures(degree=best_degree, include_bias=True).transform(x)

    # Train the Ridge regression model using the best alpha
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(x_poly, y)

    # Make predictions on player stats using the trained model
    player_poly = PolynomialFeatures(degree=best_degree, include_bias=True).transform(player_stats)
    preds = ridge_model.predict(player_poly)
