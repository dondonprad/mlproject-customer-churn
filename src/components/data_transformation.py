import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data information
        '''
        try:
            cat_col = ['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 'InternetService',
                      'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                      'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TenureMonths']
            
            num_col = ['TotalCharges', 'MonthlyCharges']

            num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                           ('scaler', StandardScaler())])
            
            cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                           ('one_hot_encoding', OneHotEncoder()),
                                           ('scaler', StandardScaler(with_mean=False))])

            logging.info(f'categorical columns: {cat_col}')
            logging.info(f'numerical columns: {num_col}')
            

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_col),
                ('cat_pipeline', cat_pipeline, cat_col)
            ])

            logging.info("Numerical Columns standard scaling completed")
            logging.info("Categorical Columns encoding completed")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
     
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('read train and test data are completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'ChurnLabel'
            num_col = ['TotalCharges', 'MonthlyCharges']

            encoders_label = {'label':LabelEncoder()}

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            #label encoder target
            encoders_label['label'].fit(target_feature_train_df)
            target_feature_train_df_new = encoders_label['label'].transform(target_feature_train_df)
            target_feature_test_df_new = encoders_label['label'].transform(target_feature_test_df)

            logging.info('Applying preprocessing objet on traning and testing dataframe')

            input_feature_training_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()

            train_arr = np.c_[input_feature_training_arr, np.array(target_feature_train_df_new)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df_new)]

            #train_arr = np.concatenate([input_feature_training_arr, np.array(target_feature_train_df)], axis=1)
            #test_arr = np.concatenate([input_feature_test_arr, np.array(target_feature_test_df)], axis=1)

            logging.info(f"Saved preprocessing object.")

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj
                )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    train_path = 'artifacts/train.csv'
    test_path = 'artifacts/test.csv'

    obj = DataTransformation()
    train_arr,test_arr,_ = obj.initiate_data_transformation(train_path,test_path)
    print(train_arr)
