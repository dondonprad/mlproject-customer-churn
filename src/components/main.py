import os
import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.data_ingestion import DataIngestion
from src.components.model_training import ModelTrainer

if __name__ == '__main__':
    
    # Load Dataset then preprocessing and split process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    # Data transformation 
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Train the model 
    model_trainer = ModelTrainer()
    model_name, result = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print('The best model is {0} with prediction acc score {1}'.format(model_name, result))
