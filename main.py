import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

data = pd.read_csv("C:\\Users\\Simo\\Desktop\\HousePrices\\train.csv")
data.drop('Id', axis=1, inplace=True)
categoricals = data.select_dtypes(exclude=[np.number])
data = pd.get_dummies(data,columns=categoricals,dummy_na=True)
data= data.fillna(0)


train, test = train_test_split(data, test_size=0.2,random_state=1)
print(train.SalePrice.describe())

print(train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()

y_truth = test["SalePrice"]
test = test.drop('SalePrice', axis=1)
features = train.drop('SalePrice', axis=1)
target = train["SalePrice"]

# print(features)
rfr = RandomForestRegressor()
rfr.fit(features,target)
y_predict = rfr.predict(test)
print(mean_squared_error(y_truth,y_predict))


