import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer,mean_squared_error

if __name__ == "__main__":
    
    df = pd.read_excel("team_stats.xlsx")
    
    ##x = df[['R','HR','RBI','SB','OBP','SLG']]
    x = df[['YEAR','R','HR','RBI','SB','OBP','SLG','K','QS','SV','ERA','WHIP','K/BB','L']]
    x = x.fillna(0)
    x = x.groupby('YEAR').transform(lambda x: (x - x.mean()) / x.std())
    x = x.fillna(0)
    print(x)

    x.to_csv("team_adjusted_stats.csv")
    x = x.to_numpy()
    
    y = df['Pts'].to_numpy()

    # Create a pipeline with PolynomialFeatures and LinearRegression
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(interaction_only=True,include_bias=True)),
        ('regression', Ridge())
    ])

    # Define the parameter grid for the polynomial degree and alpha (regularization strength)
    param_grid = {
        'poly__degree': [1,2,3,4],  # Specify the degrees to try
        'poly__interaction_only': [True,False],
        'regression__alpha': [.001,.01, .1, .25, 1.0, 2.5, 5,10.0,20,25]  # Specify the values of alpha to try
    }

    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    # Create the GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid,scoring=scoring, cv=5)

    # Fit the GridSearchCV on your data
    grid_search.fit(x, y)
    print(grid_search.scorer_)
    cvresults= pd.DataFrame(grid_search.cv_results_)
    cvresults.to_csv('cv_results.csv')
    
    # Get the best degree and alpha found by the grid search
    best_degree = grid_search.best_params_['poly__degree']
    best_alpha = grid_search.best_params_['regression__alpha']
    best_interaction = grid_search.best_params_['poly__interaction_only']
    print("best degree: ",best_degree)
    print("best alpha: ",best_alpha)
    print("best interaction: ",best_interaction)

    # Transform the data using the best degree
    x_poly = PolynomialFeatures(degree=best_degree, interaction_only=best_interaction, include_bias=True).fit_transform(x)
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(x_poly, y)
    
    # Make predictions on player stats using the trained model
    df_players = pd.read_excel("players.xlsx")
    plyrs = df_players[['R','HR','RBI','SB','OBP','SLG','SO','QS_1','SV+H','ERA','WHIP','K/BB','L']]
    print(plyrs.mean())
    print(plyrs.std())
    
    plyrs = (plyrs - plyrs.mean()) / plyrs.std()
    
    
    plyrs = plyrs.fillna(0)
    player_poly = PolynomialFeatures(degree=best_degree, interaction_only=best_interaction, include_bias=True).fit_transform(plyrs)
    
    preds = ridge_model.predict(player_poly)
    print(ridge_model.coef_)
    play_pred = pd.DataFrame()
    play_pred['Name'] = df_players['Name']
    play_pred['Scores'] = preds
    play_pred = play_pred.sort_values('Scores',ascending=False)
    print(play_pred)
    play_pred.to_excel('predictions.xlsx')
    
