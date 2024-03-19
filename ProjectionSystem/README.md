# Deep Neural Network Projection System
## Step 1: Player Performance Prediction with XGBoost
### Objective
This Python script aims to predict player's wOBA (weighted on-base average) using XGBoost, a gradient boosting algorithm. The script leverages same season underlying player data including BB%, K%, statcast data, batted ball metrics, and other plate discipline rates from fangraphs database to train the model and make predictions on players' wOBA for the given season given the set of aforementioned features.

### Functionality
#### Data Preparation:
The script reads player data from the CSV file and preprocesses it. The data includes various statistical features and a label column representing Weighted On-Base Average (wOBA).
Relevant features are selected for modeling player performance.  The dataset is split into training and testing sets using the train_test_split function from scikit-learn. This allows for the evaluation of the trained model's performance on unseen data.

#### Model Training:
Hyperparameters of the XGBoost model are tuned using RandomizedSearchCV, which searches for the best combination of hyperparameters from a predefined grid.  The hyperparameters tuned include learning rate, maximum depth, minimum child weight, subsample ratio, column subsampling ratio, and the number of estimators.  The XGBoost model is trained using the training data with the optimized hyperparameters obtained from RandomizedSearchCV.

#### Model Evaluation:
The trained model's performance is evaluated using various metrics, including R-squared score, Mean Absolute Percentage Error (MAPE), and Mean Squared Error (MSE), on the testing data.  The importance of features in predicting player performance is visualized using the plot_importance function.

## Step 2: Player Performance Prediction with GRU Model
This Python script aims to predict player performance using a GRU (Gated Recurrent Unit) model. The script leverages historical player data from a CSV file named "fangraphs_stats.csv" to train the model and make predictions.

### Objective
The primary goal of this script is to predict player performance based on various statistical features, such as walk percentage (BB%), strikeout percentage (K%), and barrel percentage (Barrel%), among others. The predictions are made for each player's performance in the upcoming season.

### Functionality
#### Data Preparation:
The script reads player data from the CSV file and preprocesses it. The data includes player names, seasons, ages, and various statistical features. For now, and as a proof of concept, only a subset of the overall features to be used are incorporated such as BB%, K%, and Barrel% are selected for modeling player performance.

#### Sequence Generation:
The player data is grouped by player name and sorted by season in ascending order. Sequences of statistical features are generated for each player.
Sequences with less than three data points (seasons) are filtered out to ensure sufficient data for training.
Sequences are padded using pad_sequences to ensure uniform length.
Additional information, such as the last season's year and age of the player, is concatenated to each sequence.
Model Training:

A sequential model is constructed using TensorFlow's Keras API.
The model architecture includes a GRU layer with 50 units followed by a Dense output layer.
Mean squared error is used as the loss function, and the model is optimized using the Adam optimizer.
Model Evaluation:

The model is trained using the prepared sequences of player data.
Training and validation losses are monitored to assess model performance.
Prediction:

Once the model is trained, it is used to predict player performance for the upcoming season.
The prediction inputs are prepared similar to the training data, and predictions are made using the trained model.
How to Use
To use this script for predicting player performance:

Ensure that the required Python libraries, including TensorFlow and pandas, are installed.
Place the player data CSV file named "fangraphs_stats.csv" in the same directory as the script.
Execute the script. It will preprocess the data, train the GRU model, and make predictions for player performance.
Note
The provided script serves as a basic example of player performance prediction using a GRU model. Depending on the specific requirements and dataset characteristics, further customization and optimization may be necessary.

## How to Use
- Ensure that the required Python libraries, including pandas, numpy, xgboost, scikit-learn, and matplotlib, are installed.
- Place the player data CSV file named "fangraphs_stats.csv" in the same directory as the script.
- Execute the script. It will preprocess the data, tune hyperparameters, train the XGBoost model, evaluate its performance, and visualize feature importance.

This README provides an overview of the script's functionality, its objectives, and instructions on how to use it. It also suggests an alternative approach to player performance prediction using a GRU model, which can be found in the provided GRU model script. Users can refer to this README to understand the purpose of the script and how to utilize it effectively for predicting player performance in fantasy baseball or similar domains


### Coming Soon......
The next steps would include leveraging the predicted underlying features for the next season found in step 2 with some known probability of error, and then leveraging these as inputs (along with associated confidence probabilities) into the trained model from step 1 to ultimately create a system given players' performances from prior years in regards to underlying (controllable) metrics such as plate discipline, batted ball data, statcast data, K%, and BB% to predict wOBA (accepted as the best non-park and non-league adjusted indicator of overall offensive performance) for an upcoming season for any given player.  
