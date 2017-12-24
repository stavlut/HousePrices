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

# data = pd.read_csv("C:\\Users\\Simo\\Desktop\\HousePrices\\train.csv")
# categoricals = data.select_dtypes(exclude=[np.number])

def numericals_data_preprocessing(data,mode):
    numericals = data.select_dtypes(include=[np.number])
    # MSSubClass - make this feature categorial
    data["MSSubClass"] = pd.Categorical(data["MSSubClass"])
    numericals = numericals.drop("MSSubClass",axis=1)
    # fill na with mean
    data['LotFrontage'].fillna(data['LotFrontage'].mean(),inplace=True)
    # fill na with 0
    data['MasVnrArea'].fillna(0,inplace=True)
    # clean outliers
    if(mode!="test"):
        data= data[data['LotFrontage'] <= 200]
        data= data[data['LotArea'] <= 60000]
        data= data[data['BsmtFinSF1'] <= 2500]
        data= data[data['GrLivArea'] <= 4000]
    for numeric in numericals:
        if(data[numeric].skew()>0.75 and numeric != "SalePrice" ):
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
    #print (column_data)
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
    return data

def Condition1_Processing(data):
    column_data=fill_na_with_value(data.Condition1, "Norm")
    column_data = column_data.replace({"PostA":"PostAB","PosN":"PostAB","RRNn":"RRNAB","RRAn":"RRNAB","RRNe":"RRNAB","RRAe":"RRNAB"})
    data["Condition1"]=column_data
    return data


def Condition2_Processing(data):
    del data["Condition2"]
    return data

def BldgType_Processing(data):
    column_data=fill_na_with_value(data.BldgType,"1Fam")
    column_data = column_data.replace({"2fmCon":"2fmDuplex","Duplex":"2fmDuplex","Twnhs":"Twnhs","TwnhsE":"Twnhs"})
    data["BldgType"]=column_data
    return data

def HouseStyle_Processing(data):
    column_data=fill_na_with_value(data.HouseStyle,"1Story")
    le = LabelEncoder()
    le.fit(column_data.unique())
    data["HouseStyle"] = le.transform(column_data)
    return data

def RoofStyle_Processing(data):
    column_data=fill_na_with_value(data.RoofStyle, "Gable")
    column_data = column_data.replace({"Flat":"roof_groop","Gambrel":"roof_groop","Mansard":"roof_groop","Shed":"roof_groop"})
    data["RoofStyle"]=column_data
    return data

def Exterior1st_Processing(data):
    column_data=fill_na_with_value(data.Exterior1st,"Other")
    data["Exterior1st"]=column_data
    return data

def Exterior2nd_Processing(data):
    column_data=fill_na_with_value(data.Exterior2nd,"Other")
    data["Exterior2nd"]=column_data
    return data

def MasVnrType_Processing(data):
    column_data=fill_na_with_value(data.MasVnrType,"None")
    data["asVnrType"]=column_data
    return data

def catrgorial_data_preprocessing(data):
    #data.drop('Id', axis=1, inplace=True)
    column_data=MSZoning_Processing(data)
    data['MSZoning'] = column_data
    #regression_check(data,categoria)
    del data["Street"]
    del data["Alley"]
    column_data=LotShape_Processing(data)
    del data['LotShape']
    data['IsRegular'] = column_data
    data['LandContour']=LandContour_Processing(data)
    del data["Utilities"]
    data["LotConfig"]=LotConfig_Processing(data)
    data=LandSlope_Processing(data)
    data=Condition1_Processing(data)
    data=Condition2_Processing(data)
    data=BldgType_Processing(data)
    data=HouseStyle_Processing(data)
    data=RoofStyle_Processing(data)
    data=Exterior1st_Processing(data)
    data=Exterior2nd_Processing(data)
    data=MasVnrType_Processing(data)
    #categoricals = data.select_dtypes(exclude=[np.number])
    #data = pd.get_dummies(data, columns=categoricals, dummy_na=True)
    return  data

def catrgorial_data_preprocessing_2(data):
    qual_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    data["ExterQual"] = data["ExterQual"].map(qual_dict).astype(int)
    data["ExterCond"] = data["ExterCond"].map(qual_dict).astype(int)
    data["BsmtQual"] = data["BsmtQual"].map(qual_dict).astype(int)
    data["BsmtCond"] = data["BsmtCond"].map(qual_dict).astype(int)
    data["HeatingQC"] = data["HeatingQC"].map(qual_dict).astype(int)
    data["KitchenQual"] = data["KitchenQual"].map(qual_dict).astype(int)
    data["FireplaceQu"] = data["FireplaceQu"].map(qual_dict).astype(int)
    data["GarageQual"] = data["GarageQual"].map(qual_dict).astype(int)
    data["GarageCond"] = data["GarageCond"].map(qual_dict).astype(int)

    data["BsmtExposure"] = data["BsmtExposure"].map(
        {None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

    bsmt_fin_dict = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    data["BsmtFinType1"] = data["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
    data["BsmtFinType2"] = data["BsmtFinType2"].map(bsmt_fin_dict).astype(int)

    data["Functional"] = data["Functional"].map(
        {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
         "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

    data["GarageFinish"] = data["GarageFinish"].map(
        {None: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)

    data["Fence"] = data["Fence"].map(
        {None: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)
    data["PoolQC"] = data["PoolQC"].map(qual_dict).astype(int)
    #
    # # IR2 and IR3 don't appear that often, so just make a distinction
    # # between regular and irregular.
    # data["IsRegularLotShape"] = (data["LotShape"] == "Reg") * 1
    # data = data.drop('LotShape', axis=1)
    # # Most properties are level; bin the other possibilities together
    # # as "not level".
    # data["IsLandLevel"] = (data["LandContour"] == "Lvl") * 1
    # data = data.drop('LandContour', axis=1)
    #
    # # Most land slopes are gentle; treat the others as "not gentle".
    # data["IsLandSlopeGentle"] = (data["LandSlope"] == "Gtl") * 1
    # data = data.drop('LandSlope', axis=1)
    #
    # # Most properties use standard circuit breakers.
    # data["IsElectricalSBrkr"] = (data["Electrical"] == "SBrkr") * 1
    # data = data.drop('Electrical', axis=1)
    #
    # # About 2/3rd have an attached garage.
    # data["IsGarageDetached"] = (data["GarageType"] == "Detchd") * 1
    # data = data.drop('GarageType', axis=1)
    #
    # # Most have a paved drive. Treat dirt/gravel and partial pavement
    # # as "not paved".
    # data["IsPavedDrive"] = (data["PavedDrive"] == "Y") * 1
    # data = data.drop('PavedDrive', axis=1)
    #
    # # The only interesting "misc. feature" is the presence of a shed.
    # data["HasShed"] = (data["MiscFeature"] == "Shed") * 1.
    # data = data.drop('MiscFeature', axis=1)
    #
    # data["HighSeason"] = data["MoSold"].replace(
    #     {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})
    # data = data.drop('MoSold', axis=1)
    #
    # data["NewerDwelling"] = data["MSSubClass"].replace(
    #     {20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
    #      90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})
    # data = data.drop('MSSubClass', axis=1)
    #
    # neighborhood_map = {
    #     "MeadowV": 0,  # 88000
    #     "IDOTRR": 1,  # 103000
    #     "BrDale": 1,  # 106000
    #     "OldTown": 1,  # 119000
    #     "Edwards": 1,  # 119500
    #     "BrkSide": 1,  # 124300
    #     "Sawyer": 1,  # 135000
    #     "Blueste": 1,  # 137500
    #     "SWISU": 2,  # 139500
    #     "NAmes": 2,  # 140000
    #     "NPkVill": 2,  # 146000
    #     "Mitchel": 2,  # 153500
    #     "SawyerW": 2,  # 179900
    #     "Gilbert": 2,  # 181000
    #     "NWAmes": 2,  # 182900
    #     "Blmngtn": 2,  # 191000
    #     "CollgCr": 2,  # 197200
    #     "ClearCr": 3,  # 200250
    #     "Crawfor": 3,  # 200624
    #     "Veenker": 3,  # 218000
    #     "Somerst": 3,  # 225500
    #     "Timber": 3,  # 228475
    #     "StoneBr": 4,  # 278000
    #     "NoRidge": 4,  # 290000
    #     "NridgHt": 4,  # 315000
    # }
    #
    # data["Neighborhood"] = data["Neighborhood"].map(neighborhood_map)
    return data

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
