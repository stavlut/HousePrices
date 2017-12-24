import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import csv
import os
import Preprocessing

def writeToFile(pred,id,file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
    results={'Id':id,'SalePrice':pred}
    with open(file_name, 'w') as csvfile:
        df=pd.DataFrame(results)
        df.to_csv(file_name, sep=',', encoding='utf-8',index=False)
    print ("done")

#plot
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

def loud_data_train_and_test():
    test= pd.read_csv("C:\\Users\\slutzky\\Desktop\\test.csv")
    train = pd.read_csv("C:\\Users\\slutzky\\Desktop\\train.csv")
    if(len(test.columns)+1==len(train.columns)):
        idtest,test=preprocess_test(test)
        target_train,train_feture=preprocess_train(train)
        return idtest,test,target_train,train_feture

# #preprocess
def preprocess_test(test):
    print("try_test1")
    print(len(test.columns))
    test_id = test['Id']
    test = test.drop('Id', axis=1)
    test = Preprocessing.numericals_data_preprocessing(test,"test")
    #test= Preprocessing.catrgorial_data_preprocessing_2(test)
    print("try_test_fillna")
    categoricals = test.select_dtypes(exclude=[np.number])
    test = pd.get_dummies(test, columns=categoricals, dummy_na=True)
    test = test.apply(lambda x: x.fillna(x.value_counts().index[0]))
    #print(test.columns)
    return test_id,test

def preprocess_train(train):
    print("try_train1")
    print(len(train.columns))
    train = train.drop('Id', axis=1)
    #train["SalePrice"] = np.log1p(train["SalePrice"])
    #train["SalePrice"] =train["SalePrice"]
    train = Preprocessing.numericals_data_preprocessing(train,"train")
    #train= Preprocessing.catrgorial_data_preprocessing_2(train)
    target=train["SalePrice"]
    train = train.drop('SalePrice', axis=1)
    print("try_train_fillna")
    categoricals = train.select_dtypes(exclude=[np.number])
    train = train.apply(lambda x: x.fillna(x.value_counts().index[0]))

    train = pd.get_dummies(train, columns=categoricals, dummy_na=True)
    #print(train.columns)
    return target,train

# categoricals = train.select_dtypes(exclude=[np.number])
# numericals = train.select_dtypes(include=[np.number])

def splite_to_train_test(train):
    train, test = train_test_split(train, test_size=0.2,random_state=3)
    print(train.SalePrice.describe())
    print(train.SalePrice.skew())
    plt.hist(train.SalePrice, color='blue')
    plt.show()
    target_test = test['SalePrice']
    target_train = train['SalePrice']
    features_test = test.drop('SalePrice', axis=1)
    features_train = train.drop('SalePrice', axis=1)
    return features_train,target_train,features_test,target_test
def lgb(features_train,target_train,features_test):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    lgb_train = lgb.Dataset(features_train, target_train)
    #lgb_eval = lgb.Dataset(test, y_truth, reference=lgb_train)
    print('Start training...')
    # train
    # gbm = lgb.train(params,lgb_train,num_boost_round=10000,valid_sets=lgb_eval)
    gbm = lgb.train(params, lgb_train, num_boost_round=10000)
    y_predlgb = gbm.predict(features_train, num_iteration=gbm.best_iteration)
    writeToFile(y_predlgb, id, "C:\\Users\\slutzky\\Desktop\\ans")



from sklearn.tree import DecisionTreeRegressor
def desTreeClass(features,target,test):
    clf = DecisionTreeRegressor()
    clf.fit(features,target)
    y_predict = clf.predict(test,check_input=True)
    return  y_predict


from sklearn.svm import LinearSVC
def LinearSVCClass(features,target,test):
    clf = LinearSVC(random_state=0)
    clf.fit(features,target)
    y_predict = clf.predict(test)
    return  y_predict


from sklearn.svm import SVC
def NonLinearSVCClass(features,target,test):
    clf = SVC(kernel="poly")
    clf.fit(features,target)
    y_predict = clf.predict(test)
    return  y_predict

def random_forest(features,target,test):
    rfr = RandomForestRegressor()
    rfr.fit(features,target)
    y_predict = rfr.predict(test)
    return y_predict

from sklearn.neighbors import KNeighborsClassifier
def KNNClass(features,target,test):
    rfr = KNeighborsClassifier(n_neighbors=3)
    rfr.fit(features,target)
    y_predict = rfr.predict(test)
    return  y_predict



def saleprice_remove_log(y_predict):
    y_predict = np.exp(y_predict)
    return y_predict
# def gbm():
#     ######## sklearn gbm  ################
#     params2 = {'n_estimators': 7200, 'max_depth': 4, 'min_samples_split': 2,
#               'learning_rate': 0.01, 'loss': 'ls'}
#     clf = GradientBoostingRegressor(**params2)
#     clf.fit(features, target)
#     y_predictgbm = clf.predict(test)
#     print()
def main():
    idtest, test, target_train, train_feture=loud_data_train_and_test()
    r=list(set(train_feture)-set(test))
    l = list( set(test)-set(train_feture))
    train_feture = train_feture.drop(r, axis=1)
    test = test.drop(l, axis=1)

    #y_predict=desTreeClass(train_feture, target_train, test)
    #y_predict=LinearSVCClass(train_feture, target_train, test)
    #y_predict=NonLinearSVCClass(train_feture, target_train, test)
    #y_predict=saleprice_remove_log(y_predict)
    #y_predict=np.exp(y_predict)
    y_predict=random_forest(train_feture, target_train, test)
    #y_predict =KNNClass(train_feture, target_train, test)
    print(y_predict)
    writeToFile(y_predict, idtest, "C:\\Users\\slutzky\\Desktop\\try\\houseprice.csv")

# print('Random Forest : The rmse of prediction is:', mean_squared_error(y_truth, y_predictrandom) ** 0.5)
#
# print('Light GBM : The rmse of prediction is:', mean_squared_error(y_truth, y_predlgb) ** 0.5)
#
# print('Sklearn GBM : The rmse of prediction is:', mean_squared_error(y_truth, y_predictgbm) ** 0.5)


main()


