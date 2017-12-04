import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import Preprocessing

data = pd.read_csv("C:\\Users\\slutzky\\Desktop\\train.csv")
data.drop('Id', axis=1, inplace=True)
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


data= Preprocessing.preprocess()


train, test = train_test_split(data, test_size=0.2,random_state=1)
print(train.SalePrice.describe())

print(train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()

y_truth = test["SalePrice"]
test = test.drop('SalePrice', axis=1)
features = train.drop('SalePrice', axis=1)
target = train["SalePrice"]

lgb_train = lgb.Dataset(features, target)
lgb_eval = lgb.Dataset(test, y_truth, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,lgb_train,num_boost_round=10000,valid_sets=lgb_eval)
y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)

####### random forest  ###########
# print(features)
rfr = RandomForestRegressor()
rfr.fit(features,target)
y_predict = rfr.predict(test)

######## sklearn gbm  ################

params2 = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = GradientBoostingRegressor(**params2)

clf.fit(features, target)
y_predict2 = clf.predict(test)

print('Random Forest : The rmse of prediction is:', mean_squared_error(y_truth, y_predict) ** 0.5)

print('Light GBM : The rmse of prediction is:', mean_squared_error(y_truth, y_pred) ** 0.5)

print('Sklearn GBM : The rmse of prediction is:', mean_squared_error(y_truth, y_predict2) ** 0.5)
