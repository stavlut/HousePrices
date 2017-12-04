import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
#import matplotlib.pyplot as plt

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


#####################################################
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


# for categoria in categoricals:
#     print("###############################################")
#     # print(categoricals)
#     print("Number of missing values: "+ str(data[categoria].isnull().sum()))
#     print(data[categoria].value_counts())
#    # print("skeweness : "+data[categoria].skew())
#     plt.style.use(style='ggplot')
#     plt.rcParams['figure.figsize'] = (10, 6)
#     print(data[categoria].describe())
#     plt.hist(data[categoria], color='blue')
#     plt.show()
#     print (categoria)
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(x=categoria, y='SalePrice', data=data)
#     xt = plt.xticks(rotation=45)

# fill missing value with a value that you choose
def fill_na_with_value(df_column,fill_na="None"):
     if fill_na is not None:
        df_column.fillna(fill_na, inplace=True)
     return df_column

# MSZoning
# CHANGE TO kategorial: {A,C}=0,{FV,I,RH}=1,{RL,RP,RM}=2
def MSZoning_Processing(data):
    column_data=fill_na_with_value(data.MSZoning,"RL")
    column_data= column_data.replace({"C (all)":"C","FV":"FV","RH":"RH_RM","RL":"RL","RM":"RH_RM"})
    return column_data

# check regretion betwin tarameters
def regression_check(df,col_name):
    plt.plot(df[col_name], df.SalePrice,'.', alpha=0.3)
    plt.legend()
    plt.show()

# LotShape
def LotShape_Processing(data):
    column_data=fill_na_with_value(data.LotShape,"Reg")
    column_data= column_data.replace({"Reg":1,"IR1":0,"IR2":0,"IR3":0 })
    print (column_data)
    return column_data

# LotShape
def LandContour_Processing(data):
    column_data=fill_na_with_value(data.LandContour,"Lvl")
    return column_data

def LotConfig_Processing(data):
    column_data=fill_na_with_value(data.LotConfig, "Inside")
    column_data= column_data.replace({"FR3":"FR","FR2":"FR"})
    return column_data

def LandSlope_Processing(data):
    column_data=fill_na_with_value(data.LandSlope,"Gtl")
    column_data = column_data.replace({"Mod": 0, "Sev": 0,"Gtl":1})
    del data["LandSlope"]
    data["ISLandSlopeGti"]=column_data

def Condition1_Processing(data):
    column_data=fill_na_with_value(data.Condition1, "Norm")
    column_data = column_data.replace({"PostA":"PostAB","PosN":"PostAB","RRNn":"RRNAB","RRAn":"RRNAB","RRNe":"RRNAB","RRAe":"RRNAB"})
    data["Condition1"]=column_data

def Condition2_Processing(data):
    del data["Condition2"]

def BldgType_Processing(data):
    column_data=fill_na_with_value(data.BldgType,"1Fam")
    column_data = column_data.replace({"2fmCon":"2fmDuplex","Duplex":"2fmDuplex","Twnhs":"Twnhs","TwnhsE":"Twnhs"})
    data["BldgType"]=column_data

def HouseStyle_Processing(data):
    column_data=fill_na_with_value(data.HouseStyle,"1Story")
    le = LabelEncoder()
    le.fit(column_data.unique())
    data["HouseStyle"] = le.transform(column_data)

def RoofStyle_Processing(data):
    column_data=fill_na_with_value(data.RoofStyle, "Gable")
    column_data = column_data.replace({"Flat":"roof_groop","Gambrel":"roof_groop","Mansard":"roof_groop","Shed":"roof_groop"})
    data["RoofStyle"]=column_data

def Exterior1st_Processing(data):
    column_data=fill_na_with_value(data.Exterior1st,"Other")
    data["Exterior1st"]=column_data

def Exterior2nd_Processing(data):
    column_data=fill_na_with_value(data.Exterior2nd,"Other")
    data["Exterior2nd"]=column_data

def MasVnrType_Processing(data):
    column_data=fill_na_with_value(data.MasVnrType,"None")
    data["asVnrType"]=column_data

def preprocess_kategory(data):
    plt.style.use(style='ggplot')
    plt.rcParams['figure.figsize'] = (10, 6)
    column_data=MSZoning_Processing(data)
    data['MSZoning'] = column_data
    #regression_check(data,categoria)
    del data["Street"]
    del data["Alley"]
    column_data=LotShape_Processing(data)
    del data['LotShape']
    data['IsRegular'] = column_data
    LandContour_Processing(data)
    del data["Utilities"]
    data["LotConfig"]=LotConfig_Processing(data)
    LandSlope_Processing(data)
    Condition1_Processing(data)
    Condition2_Processing(data)
    BldgType_Processing(data)
    HouseStyle_Processing(data)
    RoofStyle_Processing(data)
    Exterior1st_Processing(data)
    Exterior2nd_Processing(data)
    MasVnrType_Processing(data)
    return  data
#
#
# data = pd.read_csv("C:\\Users\\slutzky\\Desktop\\train.csv")
# data.drop('Id', axis=1, inplace=True)
#
# categoricals = data.select_dtypes(exclude=[np.number])
# numericals = data.select_dtypes(include=[np.number])
# #################
# del categoricals["Street"]
# del categoricals["Alley"]
# del categoricals["Utilities"]
# del categoricals['LotShape']
# del categoricals["LandSlope"]
##############

# data = pd.get_dummies(data,columns=categoricals,dummy_na=True)
