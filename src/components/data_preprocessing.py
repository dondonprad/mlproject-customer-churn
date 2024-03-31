import sys
import pandas as pd 
import numpy as np
from collections import defaultdict
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from src.logger import logging
from src.exception import CustomException


# the data loader class is class for split data frame into X(feature) and y(target)
class data_loader():
    def fit(self, df, target):
        X = df.drop(columns=target)
        y = df[target]
        return X, y

class feature_select():
    def num_cat (self, X, y):
        encoders_feature = {}
        encoders_label = {'label':LabelEncoder()}
        X_data = pd.DataFrame()
        cat_df = X.select_dtypes(include='object').columns.to_list()
        num_df = X.select_dtypes(exclude='object').columns.to_list()
        for cat in X[cat_df].columns:
            encoders_feature[cat] = LabelEncoder()
            encoders_feature[cat].fit(X[cat])
            X_data[cat] = encoders_feature[cat].transform(X[cat])
        
        X_data = pd.concat([X_data[cat_df], X[num_df]], join='outer',axis=1)

        encoders_label['label'].fit(y)
        y_data = encoders_label['label'].transform(y)

        return X_data, y_data

    def feature_name(self, X, y):

        X_data, y_data = self.num_cat(X, y)
        kbest = SelectKBest(f_classif, k=17)
        _ = kbest.fit(X_data, y_data)
        X_new = kbest.transform(X_data)
        feature = kbest.get_feature_names_out()
        return feature 
        
    def fit(self, X, y):
        feature = self.feature_name(X, y)
        X_new = X[feature]
        X_new = pd.concat([X_new, y], axis=1)
        return X_new


class split_data():
    def split(self, df):
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=43)
        return train_set, test_set


'''
#run test
class data_preprocessing():
    def __init__(self):
        self.feature_select = feature_select()
        self.data_loader = data_loader()
        self.split_data = split_data()
    
    def initiate_data_preprocessing(self):
        df = pd.read_csv('notebook\data\Telco-customer-churn.csv')
        df.drop(columns=['CustomerID','Count','Country','State','City','Zip Code','Lat Long','Latitude','Longitude','ChurnValue','ChurnReason','CLTV','ChurnScore'], inplace=True, axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
        X,y = data_loader().fit(df,'ChurnLabel')    
        new_df= feature_select().fit(X , y)
        train_set, test_set = self.split_data.split(new_df)
        return print(train_set, test_set)
    
    def run(self):
        try:
            train_set, test_set = self.initiate_data_preprocessing()
        except Exception as e:
            raise CustomException(e, sys)

if __name__=='__main__':
    obj = data_preprocessing()
    obj.run()
'''