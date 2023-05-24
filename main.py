import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    
    df = pd.read_csv("team_stats.csv")
    
    ##x = df[['R','HR','RBI','SB','OBP','SLG']]
    x = df[['YEAR','R','HR','RBI','SB','OBP','SLG','K','QS','SV','ERA','WHIP','K/BB','L']]
    x = x.fillna(0)
    Mean = x.groupby(['YEAR']).mean()
    print(Mean)
    stdDev = x.groupby(['YEAR']).std()
    print(stdDev)
    years = x.YEAR
    x.drop(['YEAR'],axis=1,inplace=True)
    # HR_mean = x.groupby(['YEAR']).HR.mean()
    # RBI_Mean = x.groupby(['YEAR']).RBI.mean()
    # SB_Mean = x.groupby(['YEAR']).SB.mean()
    # OBP_Mean = x.groupby(['YEAR']).OBP.mean()
    # SLG_Mean = x.groupby(['YEAR']).SLG.mean()
    
    
    x = (x-Mean)/stdDev
    x.to_csv("team_adjusted_stats.csv")
    x = x.to_numpy()
    y = df['Pts'].to_numpy()

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
    x_poly = PolynomialFeatures(degree=best_degree, include_bias=True).fit_transform(x)

    # Train the Ridge regression model using the best alpha
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(x_poly, y)

    # Make predictions on player stats using the trained model
    df_players = pd.read_csv("team_stats.csv")
    
    plyrs = df_players[['R','HR','RBI','SB','OBP','SLG','K','QS','SV','ERA','WHIP','K/BB','L']]
    plyrs.fillna(0)
    player_poly = PolynomialFeatures(degree=best_degree, include_bias=True).transform(plyrs)
    preds = ridge_model.predict(player_poly)
    print(preds)
