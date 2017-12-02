import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

data = pd.read_csv("C:\\Users\\Simo\\Desktop\\HousePrices\\train.csv")
data.drop('Id', axis=1, inplace=True)
categoricals = data.select_dtypes(exclude=[np.number])
data = pd.get_dummies(data,columns=categoricals,dummy_na=True)
data= data.fillna(0)