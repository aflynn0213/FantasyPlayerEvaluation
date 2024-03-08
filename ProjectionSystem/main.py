import pandas as pd
import numpy as np
from xgboost import XGBRegressor as xgbr

from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error,mean_squared_error

from xgboost import plot_importance
import matplotlib.pyplot as plt

if __name__ == "__main__":

    players_data = pd.read_csv('fangraphs_stats.csv')
    players_data.columns = players_data.columns.str.lower()
    print(players_data)

    #FIGURE OUT FEATURES AND LABELS
    simple_feats = players_data.iloc[:,[10,11,12,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,50,52,53]]
    print(simple_feats.columns)
    woba_label = players_data.loc[:,"woba"]
    
    xtr,xte,ytr,yte = train_test_split(simple_feats,woba_label,test_size=.2,random_state=5)
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 2, 3, 4],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'n_estimators': [100, 200, 300, 400]
    }

    # Create RandomizedSearchCV object
    cv_model = RandomizedSearchCV(
        xgbr(), param_distributions=param_grid, n_iter=10, cv=5, verbose=2, n_jobs=-1
    )
    #RUNS RANDOMIZED HYPERPARAMETER TUNING
    cv_model.fit(xtr,ytr)
    print(cv_model.best_params_)
    print(cv_model.best_score_)

    woba_preds = cv_model.best_estimator_.predict(xte)
    print("R2 SCORE: ", r2_score(yte,woba_preds))
    print("MAE: ",mean_absolute_percentage_error(yte, woba_preds))
    print("MSE: ", mean_squared_error(yte, woba_preds))

    plot_importance(cv_model.best_estimator_,importance_type='gain')
    plt.show()