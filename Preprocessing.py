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
def numericals_data_preprocessing(data):
    numericals = data.select_dtypes(include=[np.number])
    # MSSubClass - make this feature categorial
    data["MSSubClass"] = pd.Categorical(data["MSSubClass"])
    numericals = numericals.drop("MSSubClass",axis=1)
    # fill na with mean
    data['LotFrontage'].fillna(data['LotFrontage'].mean(),inplace=True)
    # fill na with 0
    data['MasVnrArea'].fillna(0,inplace=True)

    # clean outliers
    data= data[data['LotFrontage'] <= 200]
    data= data[data['LotArea'] <= 60000]
    data= data[data['BsmtFinSF1'] <= 2500]
    data= data[data['GrLivArea'] <= 4000]
    for numeric in numericals:
        if(data[numeric].skew()>0.75):
            data[numeric] = np.log1p(data[numeric])


    # remove features
    data.drop('BsmtFinSF2',axis=1, inplace=True)
    data.drop('LowQualFinSF',axis=1, inplace=True)
    data.drop('3SsnPorch',axis=1, inplace=True)
    data.drop('ScreenPorch',axis=1, inplace=True)
    data.drop('PoolArea',axis=1, inplace=True)
    data.drop('MiscVal',axis=1, inplace=True)

    return data

# print(data['LotArea'].isnull().sum())


# for numeric in numericals:
#     print("###############################################")
#     print(numeric)
#     print("Number of missing values: "+ str(data[numeric].isnull().sum()))
#     print(data[numeric].value_counts())
#     data[numeric] = data[numeric].fillna(0)
#     print("skeweness : "+str(data[numeric].skew()))
#     print(data[numeric].describe())
#     plt.hist(data[numeric], color='blue')
#     plt.show()


# data = pd.get_dummies(data,columns=categoricals,dummy_na=True)
