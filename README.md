# FantasyPlayerEvaluation

This repository is broken up into different projects each of which is described briefly below:

## LeagueHistoryLinearWeightsModel
Contained within this directory is a main.py python script is designed for fantasy baseball analysis, particularly tailored to my Head-to-Head (H2H) Each Category scoring league.  The main execution block of the script revolves around leveraging historical league data, stored in an Excel file named "team_stats.xlsx," to predict team performance based on various statistical categories. The data preprocessing phase involves selecting relevant features and standardizing them using z-score normalization, essential for ensuring uniformity across different statistical metrics, such as runs, home runs, RBIs, rate stats (OBP, SLUG, ERA, etc.) and more. Through the utilization of machine learning techniques, particularly PolynomialFeatures and Ridge Regression, the script constructs a predictive model optimized for mean squared error. This model undergoes hyperparameter tuning via GridSearchCV to identify the optimal combination of parameters for accurate predictions. Subsequently, player projections from another Excel file are processed similarly, generating z-scores for each player's statistical categories. The model coefficients learned from the team data are then applied to these z-scored player statistics to predict team points based on the individual contributions of each player to their respective categories. Finally, the script outputs the predictions, along with z-scores of players, facilitating further analysis and decision-making in fantasy baseball management. This comprehensive approach not only provides insights into team performance prediction but also aids in strategic player selection and team optimization within the fantasy baseball framework.

#### Data Preparation:
The historical league data from "team_stats.xlsx" contains statistics of various teams across different seasons. These statistics include metrics such as runs, home runs, RBIs, stolen bases, on-base percentage (OBP), slugging percentage (SLG), strikeouts (K), quality starts (QS), saves (SV), earned run average (ERA), walks plus hits per inning pitched (WHIP), strikeouts to walk ratio (K/BB), and losses (L).
To ensure uniformity across these metrics, z-score normalization is applied, transforming each statistic into a dimensionless value representing its deviation from the mean in terms of standard deviations. This normalization is crucial because different statistical categories may have vastly different scales.

#### Model Construction:
A predictive model is constructed using PolynomialFeatures and Ridge Regression. PolynomialFeatures generates polynomial and interaction features, allowing the model to capture potential nonlinear relationships between input features. Ridge Regression introduces regularization to the linear regression model, preventing overfitting by penalizing large coefficients.
The target variable for the model is the total points earned by each team in the fantasy league. Points are awarded based on the outcome of matchups, with wins yielding 2 points and ties yielding 1 point.

#### Training the Model:
The constructed model is trained on the z-score normalized team statistics, with the target variable being the total points earned by each team. This training phase involves optimizing the model's parameters, including the degree of polynomial features and the regularization strength, through GridSearchCV.

#### Learning Coefficients:
Once the model is trained, the coefficients associated with each feature (category z-score) are learned. These coefficients represent the relative importance or contribution of each statistical category to the overall team performance in terms of points.

#### Predicting Team Points:
To predict the points earned by a team, the z-scores of statistical categories for that team are calculated and multiplied by the corresponding coefficients learned from the model. This process effectively assigns a weighted value to each category based on its impact on team performance.
These weighted z-scores are then aggregated to obtain a predicted point value for the team. This prediction reflects the team's expected performance in the fantasy league based on its statistical profile.

#### Player Point Assignment:
Similarly, player projections are processed to derive z-scores for each statistical category. These z-scores represent the performance of individual players relative to the league average in each category.
The coefficients learned from the team model are then applied to these player z-scores. By multiplying each player's z-score by the corresponding coefficient for the respective category, a point value is assigned to the player based on their expected contribution to team performance in that category.
