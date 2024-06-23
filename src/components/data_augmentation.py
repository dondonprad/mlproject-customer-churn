import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException


class SMOTEAugmentation():
    def fit_resample(self, train_array): #, label):
        try: 
            logging.info("Data Augmentation using SMOTE Initiated")
            X_train, y_train = (train_array[:,:-1],
                                train_array[:,-1])

            aug = SMOTE(sampling_strategy='minority', random_state=43)
            X_over, y_over = aug.fit_resample(X_train, y_train)
            smote_train_arr = np.c_[X_over, y_over]
            logging.info("Data Augmentation using SMOTE Successful")
            return smote_train_arr
        
        except Exception as e:
            raise CustomException(e, sys)
        

class DataProportionAugmentation():
    def data_sampler(self, data, no_new_data, preserve):
        X_new=[]
        data_gen=[]
        disc_data= list(set(data))

        for i in disc_data:
            x=data.count(i)
            X_new.append(round(x*(no_new_data) / len(data)))

        for j in range(0,len(X_new)):
            for i in range(X_new[j]):
                data_gen.append(disc_data[j])

        if(len(data_gen)==0):
            data_gen = data + data_gen

        if(no_new_data > sum(X_new)):
            data_gen = data + data_gen
            data_gen = list(data_gen + list(np.random.choice(data_gen,int(no_new_data-sum(X_new)),replace = False)))

        data_gen = list(np.random.choice(data_gen,int(no_new_data),replace = False))

        if(preserve == True):
            data_gen = data + data_gen

        return data_gen
    
    def label_split(self, df_class, diff, label_name, label_column, preserve):
        df_ = pd.DataFrame(columns=[])
        del df_class[label_column]
        for (columnName, columnData) in df_class.iteritems():
           df_[columnName] = self.data_sampler(list(columnData.values), diff, preserve)
        df_[label_column] = label_name
        return df_
    
    def class_balance(self, data, label_column, preserve = True):
        split_list=[]
        for label, df_label in data.groupby(label_column):
            split_list.append(df_label)

        maxLength = max(len(x) for x in split_list)
        augmented_list=[]
        for i in range(0,len(split_list)):
            label_name = list(set(split_list[i][label_column]))[0]
            diff = maxLength - len(split_list[i])
            augmented_list.append(self.label_split(split_list[i], diff, label_name, label_column, preserve))
        finaldf = pd.DataFrame(columns=[])


        for i in range(0,len(split_list)):
            finaldf = pd.concat([finaldf,augmented_list[i]],axis=0)

        return finaldf
    
    def fit_resemple(self, X_train, y_train, label):
        try:
            logging.info("Data Augmentation using Data Prportion Initiated")
            y_train = pd.DataFrame(data=y_train,columns=[label])
            df = pd.concat([X_train, y_train], join='outer', axis=1)
            a_df = self.class_balance(data=df,label_column=label, augment = False, preserve = True)
            a_df.to_csv(self.data_proportion_augmentation_config.data_proportion_augmentation_path, index=False, header=True)
            
            logging.info("Data Augmentation using Data Prportion Successful")
            return self.data_proportion_augmentation_config.data_proportion_augmentation_path
            
        except Exception as e:
            raise CustomException(e, sys)




    
    