import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,make_scorer, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def loadInputData(test=False):
    if (test):
        data = pd.read_csv('testing2024_hitters.csv')
        # Assuming your data is already cleaned and preprocessed
        data.set_index([data.columns[0], data.columns[1]], inplace=True)
        # Split the data into features and labels
        X = data.iloc[:, 1:-7]  # Features (columns B through AE)
        y = data.iloc[:, -7:-4]   # Labels (columns AF, AG, AH)
    else:
        data = pd.read_csv('training_hitters.csv')
        # Assuming your data is already cleaned and preprocessed
        data.set_index([data.columns[0], data.columns[1]], inplace=True)
        # Split the data into features and labels
        X = data.iloc[:, 1:-6]  # Features (columns B through AE)
        y = data.iloc[:, -6:-3]   # Labels (columns AF, AG, AH)
    
    return X,y,data

def transformX(x,SS):#,PC):
    return SS.transform(x)
    

if __name__ == "__main__": 
    # Load your dataset, assuming it's a CSV file
    X,y,data = loadInputData()
    
    # Split the data into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    std_sc = StandardScaler().fit(X)
    X = std_sc.transform(X)
    
    # Init PCA object
    pca = PCA()
    # Fit the PCA to your data
    pca.fit(X)
    # Plot cumulative explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.show()
    
    # Object to be used
    #pca_obj = PCA(n_components=10).fit(X)
    #x_pca = pca_obj.transform(X)
    
    # Initialize dictionaries to store models and their predictions
    models = {}
    predictions = {}
    polys = {}
    ridges = {}
    feats = {}
    #scoring = make_scorer(r2_score)
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    #     }
    
    # rf = RandomForestRegressor(random_state=42)
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    # # Train a separate model for each output using RandomForestRegressor
    # print("HERE")
    # for column in y.columns:  
        
    #     # Instantiate GridSearchCV
    #     grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring=scoring, n_jobs=-1)
        
    #     # Perform the grid search
    #     grid_search_rf.fit(X, y[column])
        
    #     # Print the best parameters found
    #     print("Best Parameters:", grid_search_rf.best_params_)
        
    #     # Make predictions using the best model
    #     best_model_rf = grid_search_rf.best_estimator_
    #     y_pred_rf = best_model_rf.predict(X)
    #     models[f'{column}'] = best_model_rf
    #     predictions[f'{column}'] = y_pred_rf
    #     feats_value = best_model_rf.feature_importances_
    #     feats[f'{column}'] = np.argsort(feats_value)[::-1]
        
    # print("Feature ranking:")
    # for col,features in feats_value:
    #     for i, idx in enumerate(features):
    #         print(f"{i + 1}. Feature {idx}: {feature_importance[idx]}")
        
    #     # Plot feature importance
    #     plt.figure(figsize=(10, 6))
    #     plt.title("Feature Importance")
    #     plt.bar(range(X.shape[1]), feature_importance[features],
    #            color="r", align="center")
    #     plt.xticks(range(X.shape[1]), indices)
    #     plt.xlabel("Feature Index")
    #     plt.ylabel("Feature Importance")
    #     plt.show()
        
    pipeline = Pipeline([
            ('poly', PolynomialFeatures(include_bias=True)),
            ('ridge', Lasso())
        ])
    param_grid = {
            'poly__degree': [1,2,3,4],  # Specify the degrees to try
            'poly__interaction_only': [False],
            'ridge__alpha': [.001,.01, .1, .25, 1.0, 2.5, 5,10.0,20]  # Specify the values of alpha to try
        }
    
    
    #Train a separate model for each output using Polynomial Regression
    for column in y.columns:
    
        # GridSearchCV object
        grid_search_lr = GridSearchCV(pipeline, param_grid,scoring=scoring, n_jobs=-1,cv=5)
        # GridSearchCV fit runs the gridsearch algorithm
        #grid_search_lr.fit(X_train, y_train[column])
        grid_search_lr.fit(X, y[column])
        
        best_estimator = grid_search_lr.best_estimator_
        poly = best_estimator.named_steps['poly']
        ridge_model = best_estimator.named_steps['ridge'] 
        print("BEST SCORE: ",grid_search_lr.best_score_)
        # Transforms the team data using the best estimator and fits to team wins
        x_poly = poly.transform(X)
        ridge_model.fit(x_poly, y[column])
        models[f'Polynomial_{column}'] = pipeline
        polys[f'{column}'] = poly
        ridges[f'{column}'] = ridge_model
        predictions[f'Polynomial_{column}'] = ridge_model.predict(x_poly)
        print(ridge_model.coef_)

    
    for col , model in models.items():
        print(f"{col} best training paramaters: ",model.get_params() )
    
    
    # # Evaluate the models
    # mse_results = {}
    # print(y_test)
    # print(predictions)
    # for name, model in models.items():
    #     mse = mean_squared_error(y_test[name.split('_')[1]], predictions[name])
    #     mse_results[name] = mse
    #     print(f"Mean Squared Error ({name}): {mse}")
    
        
    # perc_err = {}
    # for name,model in models.items():
    #     perc_error = abs((y_test[name.split('_')[1]] - predictions[name]) / y_test[name.split('_')[1]]) * 100
    #     # Calculate average percentage error
    #     avg = perc_error.mean()
    #     perc_err[name] = avg
    #     print(f"Average Percentage Error ({name}): {avg}")
    
        
    # # Choose the model with the lowest error percentage for each output
    # best_models = {}
    # for column in y_train.columns:
    #     best_model_name = min((name for name in perc_err.keys() if name.startswith(('RandomForest', 'Polynomial')) and name.endswith(column)), key=perc_err.get)
    #     best_models[column] = models[best_model_name]
    
    # #Print the best model for each output
    # print("\nBest Models:")
    # for column, model in best_models.items():
    #     print(f"Best Model for {column}: {model}")
    
    
    X_test,y,test_data = loadInputData(True)
    test_preds = {}
    mse = {}
    percentage_error = {}
    r2 = {}
    
    X_test = transformX(X_test,std_sc)#,pca_obj)
    ########## POLYNOMIAL #########################
    for column in y.columns:
        x_poly = polys[column].transform(X_test)
        test_preds[column] = ridges[column].predict(x_poly)
        mse[column] = mean_squared_error(y[column],test_preds[column])
        tmp_perc = abs(( y[column] - test_preds[column]) / y[column]) * 100
        percentage_error[column] = tmp_perc.mean()
        r2[column] = r2_score(y[column],test_preds[column])
        
    ############## RANDOM FOREST ##############################
    # for column in y.columns:
    #     test_preds[column] = models[column].predict(X_test)
    #     mse[column] = mean_squared_error(y[column],test_preds[column])
    #     tmp_perc = abs(( y[column] - test_preds[column]) / y[column]) * 100
    #     percentage_error[column] = tmp_perc.mean()
    #     r2[column] = r2_score(y[column],test_preds[column])
        
    print(test_preds["wOBA"])
    xwoba_act = test_data["xwOBA"]
    comp_mse = mean_squared_error(xwoba_act,test_preds["wOBA"])
    comp_perc = abs(( xwoba_act - test_preds["wOBA"]) / xwoba_act) * 100
    comp_perc=comp_perc.mean()
    
    for col in y.columns:
        print(f"MSE of {col}: ",mse[col])
        print(f"Error % of {col}: ",percentage_error[col])
        print(f"R2 of {col}: ",r2[col])
        
    print("MSE of xwOBA from statcast to my algorithm: ", comp_mse)
    print("Error% of xwOBA statcast from my algorithm: ", comp_perc)
    print("R2 of xwoba statcast and my algorithm: ", r2_score(xwoba_act,test_preds["wOBA"]))
    
    players_predicted_wobas = pd.DataFrame([[test_preds["wOBA"],y["wOBA"],test_data["xwOBA"]]],index=test_data.index)
    players_predicted_wobas.to_csv("players_pred_wOBA.csv")
    
    
