import os
import sys
from src.logger import logging
from src.exception import CustomException
from data_preprocessing import data_loader
from data_preprocessing import feature_select
from data_preprocessing import split_data
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass 

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('notebook\data\Telco-customer-churn.csv')
            logging.info('Dataset ready to use')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.drop(columns=['CustomerID','Count','Country','State','City','Zip Code','Lat Long','Latitude','Longitude','ChurnValue','ChurnReason','CLTV','ChurnScore'], inplace=True, axis=1)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)
            X,y = data_loader().fit(df,'ChurnLabel')
            new_df = feature_select().fit(X , y)

            #train test split 
            logging.info('Train test split initiated')
            train_set, test_set = split_data().split(new_df)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion Data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,

            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)