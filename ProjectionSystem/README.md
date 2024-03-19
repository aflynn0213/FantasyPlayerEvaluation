# Deep Neural Network Projection System

## Step 1: Player wOBA within Same Season Prediction with XGBoost

### Objective
This Python script aims to predict players' wOBA (weighted on-base average) using XGBoost, a gradient boosting algorithm. Leveraging same-season underlying player data, including BB%, K%, statcast data, batted ball metrics, and other plate discipline rates from the Fangraphs database, the model predicts players' wOBA for the given season based on the set of aforementioned features.

### Functionality

#### Data Preparation
The script reads player data from the CSV file and preprocesses it, selecting various statistical features and a label column representing wOBA. The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn to evaluate the trained model's performance on unseen data.

#### Model Training
Hyperparameters of the XGBoost model are tuned using RandomizedSearchCV, searching for the best combination of hyperparameters from a predefined grid. Hyperparameters include learning rate, maximum depth, minimum child weight, subsample ratio, column subsampling ratio, and the number of estimators. The XGBoost model is then trained using the training data with optimized hyperparameters.

#### Model Evaluation
The trained model's performance is evaluated using various metrics, including R-squared score, Mean Absolute Percentage Error (MAPE), and Mean Squared Error (MSE), on the testing data. Feature importance in predicting player performance is visualized using the `plot_importance` function.

### How to Use
- Ensure that the required Python libraries, including pandas, numpy, xgboost, scikit-learn, and matplotlib, are installed.
- Place the player data CSV file named "fangraphs_stats.csv" in the same directory as the script.
- Execute the script. It preprocesses the data, tunes hyperparameters, trains the XGBoost model, evaluates its performance, and visualizes feature importance.

## Step 2: Player Performance Prediction with GRU Model

### Objective
This Python script aims to predict player performance using a GRU (Gated Recurrent Unit) model. Historical player data from the "fangraphs_stats.csv" CSV file is leveraged to train the model and make predictions.

### Functionality

#### Data Preparation
The script reads player data from the CSV file and preprocesses it, including player names, seasons, ages, and various statistical features. A subset of features, such as BB%, K%, and Barrel%, is selected for modeling player performance.

#### Sequence Generation
Player data is grouped by player name, sorted by season in ascending order, and sequences of statistical features are generated for each player. Sequences with fewer than three data points (seasons) are filtered out to ensure sufficient data for training. Sequences are padded using `pad_sequences`, and additional information such as the last season's year and age of the player is concatenated to each sequence.

#### Model Training
A sequential model is constructed using TensorFlow's Keras API, including a GRU layer with 50 units followed by a Dense output layer. Mean squared error is used as the loss function, and the model is optimized using the Adam optimizer. The model is trained using the prepared sequences of player data, with training and validation losses monitored to assess model performance.

#### Prediction
Once the model is trained, it predicts player performance for the upcoming season. Prediction inputs are prepared similar to the training data, and predictions are made using the trained model.

### How to Use
To predict player performance:
- Ensure TensorFlow and pandas are installed.
- Place "fangraphs_stats.csv" in the script's directory.
- Execute the script to preprocess data, train the GRU model, and make predictions.

### Note
This script serves as a basic example of player performance prediction using a GRU model. Depending on specific requirements and dataset characteristics, further customization and optimization may be necessary.

### Coming Soon
The next steps involve leveraging the predicted underlying metrics for the next season from Step 2, incorporating known probabilities of error on these predictions, and using these as inputs (along with their associated confidence probabilities) into the trained model from Step 1. This ideally will create a system to utlimately predict wOBA for an upcoming season based on a given player's performances from prior years in regards to the underlying metrics discussed throughout and ones that more accurately define a player's true talent.
