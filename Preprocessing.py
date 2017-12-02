import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

data = pd.read_csv("C:\\Users\\Simo\\Desktop\\HousePrices\\train.csv")
data.drop('Id', axis=1, inplace=True)

categoricals = data.select_dtypes(exclude=[np.number])
numericals = data.select_dtypes(include=[np.number])
# data = data.fillna(0)

for numeric in numericals:
    print("###############################################")
    print(numeric)
    print("Number of missing values: "+ str(data[numeric].isnull().sum()))
    print(data[numeric].value_counts())
    data[numeric] = data[numeric].fillna(0)
    print("skeweness : "+str(data[numeric].skew()))
    print(data[numeric].describe())
    plt.hist(data[numeric], color='blue')
    plt.show()


# data = pd.get_dummies(data,columns=categoricals,dummy_na=True)
