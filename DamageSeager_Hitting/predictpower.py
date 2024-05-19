import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline

def normalize_names(names):
    normalized_names = []
    for name in names:
        # Replace special characters
        name = name.replace("Ã¡", "a").replace("Ã©", "e").replace("Ã­", "i").replace("Ã³", "o").replace("Ãº", "u").replace("Ã±", "n").replace("â€™", "'").replace("â€˜", "'").replace("Ã¡", "a").replace("Ã©", "e").replace("Ã­", "i").replace("Ã³", "o").replace("Ãº", "u").replace("Ã±", "n").replace("â€™", "'").replace("â€˜", "'")
        # Capitalize each word
        name = ' '.join(word.capitalize() for word in name.split())
        normalized_names.append(name)
    return normalized_names

if __name__ == "__main__":
    file_path = 'Dam_Seag_Leaders.xlsm'


    df = pd.read_excel(file_path,sheet_name='LR')
    df.columns = df.columns.str.lower()
    df.set_index(['year','name'],inplace=True)
    cols =["woba","90th pctile ev","damage/bbe (%)","seager","pulled fb (%)","selectivity (%)","hittable pitch take (%)","chase (%)", "z-contact (%)"]
    df = df[cols]
    x_train = df.iloc[:,1:]
    y_train = df["woba"]

    print(x_train)
    print(y_train)
    pipeline = Pipeline([
            ('poly', PolynomialFeatures(include_bias=True)),
            ('regressor', Lasso())
        ])
    param_grid = {
            'regressor': [Lasso(),Ridge()],
            'poly__degree': [1,2,3,4],  # Specify the degrees to try
            'poly__interaction_only': [False],
            'poly__include_bias': [True,False],
            'regressor__alpha': [.001,.01, .1, .25, 1.0, 2.5, 5,10.0,20]  # Specify the values of alpha to try
        }

    scorer = make_scorer(r2_score)
    # GridSearchCV object
    model = GridSearchCV(pipeline, param_grid,scoring=scorer, n_jobs=-1,cv=4)
    # GridSearchCV fit runs the gridsearch algorithm
    model.fit(x_train,y_train)

    best_estimator = model.best_estimator_
    best_regressor = best_estimator.named_steps['regressor']

    if isinstance(best_regressor, Lasso):
        print("Best regressor: Lasso")
    elif isinstance(best_regressor, Ridge):
        print("Best regressor: Ridge")
    else:
        print("Unknown best regressor")

    poly = best_estimator.named_steps['poly']
    regress = best_estimator.named_steps['regressor']
    
    # Get regression coefficients
    coefficients = regress.coef_

    # Get the feature names
    interaction_terms = poly.get_feature_names_out()

    # Map coefficients to feature names
    coefficients_map = dict(zip(interaction_terms, coefficients))

    # Print coefficients for each feature
    for feature, coefficient in coefficients_map.items():
        if(coefficient != 0):
            print(f"{feature}: {coefficient}")

    print(f"best training parameters: ", regress.get_params())
    print(f"best poly training parameters: ", poly.get_params())
    print("BEST SCORE: ",model.best_score_)


    # Specify the name or index of the sheet you want to read
    sheet_name = '2021'  # Replace 'Sheet1' with the name of your sheet, or use an index (e.g., 0, 1, 2, ...)

    # Read the specific sheet into a DataFrame
    df2021 = pd.read_excel(file_path, sheet_name=sheet_name)
    df2022 = pd.read_excel(file_path,sheet_name='2022')
    df2023 = pd.read_excel(file_path,sheet_name='2023')
    
    df2021.columns = df2021.columns.str.lower()
    df2022.columns = df2022.columns.str.lower()
    df2023.columns = df2023.columns.str.lower()
    df2021.set_index(['season','name'],inplace=True)
    df2022.set_index(['season','name'],inplace=True)
    df2023.set_index(['season','name'],inplace=True)

    df = pd.concat([df2021,df2022,df2023])
    players_2021 = df.index.get_level_values('name')[df.index.get_level_values('season') == 2021].unique()
    players_2022 = df.index.get_level_values('name')[df.index.get_level_values('season') == 2022].unique()
    players_2023 = df.index.get_level_values('name')[df.index.get_level_values('season') == 2023].unique()
    
    common_players = set(players_2021).intersection(players_2022)
    common_players_2 = set(players_2022).intersection(players_2023)
    
    # Step 2: Filter the data for common players from 2021 and 2022
    x_1 = df.loc[(2021, list(common_players)), "xwoba"]
    y_1 = df.loc[(2022, list(common_players)), "woba"]
    x_2 = df.loc[(2022, list(common_players_2)), "xwoba"]
    y_2 = df.loc[(2023, list(common_players_2)), "woba"]
    x_1 = x_1.sort_index(level='name')
    y_1 = y_1.sort_index(level='name')
    x_2 = x_2.sort_index(level='name')
    y_2 = y_2.sort_index(level='name')
    x = pd.concat([x_1,x_2])
    y = pd.concat([y_1,y_2])

    print(x)
    print(y)

    print("R2 SCORE OF xwOBA to next year wOBA: ", r2_score(y,x))

    df.columns = df.columns.str.strip()
    # 
    cols =["90th pctile ev","damage/bbe (%)","seager","pulled fb (%)","selectivity (%)","hittable pitch take (%)","chase (%)", "z-contact (%)"]
    cols = [col.strip() for col in cols]
    y_test = df["woba"]
    x_test = df[cols]

    print(y_test)
    print(x_test)
    

    # Transforms the team data using the best estimator and fits to team wins
    x_poly = poly.transform(x_test)
    preds = regress.predict(x_poly)
    print("R2 SCORE model for next year wOBA: ", r2_score(y_test,preds))

    
    
    df_2024 = pd.read_excel(file_path,sheet_name='2024')
    df_2024.columns = df_2024.columns.str.lower()
    df_2024.set_index(['season','name'],inplace=True)
    cols =["90th pctile ev", "damage/bbe (%)","seager","pulled fb (%)","selectivity (%)","hittable pitch take (%)","chase (%)", "z-contact (%)"]
    x_2024 = df_2024[cols]
    #woba_2024 = df_2024["woba"] 

    x24_poly = poly.transform(x_2024)
    preds = regress.predict(x24_poly)
    #print("R2 of model for wOBA: ", r2_score(woba_2024,preds))
    df_2024["py woba"] = preds
    sorted_df = df_2024.sort_values(by='py woba', ascending=False)
    # Print the top 50 rows
    print(sorted_df["py woba"].head(50))
    df_2024.to_excel('python_outputs.xlsx', index=True)





