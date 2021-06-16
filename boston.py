import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#load the boston dataset from Sci-Kit Learn API
df = load_boston()
x = pd.DataFrame(df.data, columns=df.feature_names)
y = pd.Series(df.target)

#Check the first five rows of the data
print(x.head())

#Create the training and test sets as 20% and 80%
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=123)

#Initialize the XGBoost
#Note that these parameters can change by the dataset, you can take a quick look to the the xgboost parameters documentation for more information 
regressor = xgb.XGBRegressor(
    n_estimators=100,
    reg_lambda=1,
    gamma=0,
    max_depth=3
)

#Fit the model to the regressor that defined above
regressor.fit(x_train, y_train)

#Now predict a house price according to the given dataset
y_pred = regressor.predict(x_test)

#Calculate mean squared error to see how good the model performs
print(mean_squared_error(y_test, y_pred))