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
    #Put zeros in for the years where the stats weren't valid, the model should be able to handle
    #learning new stats even if they were previously zero'd out
    x = x.fillna(0)
    x = x.groupby('YEAR').transform(lambda x: (x - x.mean()) / x.std())
    #This gets rid of nan's
    x = x.fillna(0)
    print(x)
    #Write z score adjusted team stats to csv for monitoring
    x.to_csv("team_adjusted_stats.csv")
    x = x.to_numpy()
    
    y = df['Pts'].to_numpy()

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

    #Debug purposes writing results from folds for each combo to csv
    cvresults= pd.DataFrame(grid_search.cv_results_)
    cvresults.to_csv('cv_results.csv')
    
    # Best hyperparamters found by the grid search
    print("best params: ", grid_search.best_params_)

    # Best estimator which is a pipeline type from the grid search object
    #Followed by the specific steps of the pipeline
    best_estimator = grid_search.best_estimator_
    poly = best_estimator.named_steps['poly']
    ridge_model = best_estimator.named_steps['ridge']

    # Transforms the team data using the best estimator and fits to team wins
    x_poly = poly.transform(x)
    ridge_model.fit(x_poly, y)
    
    # Read in player stats
    df_players = pd.read_excel("players.xlsx")
    plyrs = df_players[['R','HR','RBI','SB','OBP','SLG','SO','QS_1','SV+H','ERA','WHIP','K/BB','L']]
    #Debug statements
    print(plyrs.mean())
    print(plyrs.std())
    #For SP we leave Saves+Holds blank so they're not used in this calculation and the zeros don't bring down the mean
    #Likewise with RP we leave QS blank so they don't get used in the calculation below and inflate the z scores for starting pitchers
    #by bringing the mean for QS much lower than is reflective of the actual data
    plyrs = (plyrs - plyrs.mean()) / plyrs.std()
    #Fill in Zeros now for the missing stats so that linear regression can use this in predictions (zeros have no affect)
    plyrs = plyrs.fillna(0)
    #Applies same transformation as was done to team stats but to players (test) data
    player_poly = poly.transform(plyrs)
    # Make predictions using the ridge model
    preds = ridge_model.predict(player_poly)

    # Create a DataFrame to store the predictions
    play_pred = pd.DataFrame()
    play_pred['Name'] = df_players['Name']
    play_pred['Scores'] = preds
    play_pred = play_pred.sort_values('Scores',ascending=False)
    print(play_pred)
    play_pred.to_excel('predictions.xlsx')
    
