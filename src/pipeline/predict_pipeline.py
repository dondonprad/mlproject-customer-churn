import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try: 
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 SeniorCitizen:str,
                 Partner:str,
                 Dependents:str,
                 MultipleLines:str,
                 InternetService:str,
                 OnlineSecurity:str,
                 OnlineBackup:str,
                 DeviceProtection:str,
                 TechSupport:str,
                 StreamingTV:str,
                 StreamingMovies:str,
                 Contract:str,
                 PaperlessBilling:str,
                 PaymentMethod:str,
                 TenureMonths:int,
                 MonthlyCharges:float,
                 TotalCharges:float
                 ):
        
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup,
        self.DeviceProtection = DeviceProtection,
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies,
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling,
        self.PaymentMethod = PaymentMethod
        self.TenureMonths = TenureMonths
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'SeniorCitizen':[self.SeniorCitizen],
                'Partner':[self.Partner],
                'Dependents':[self.Dependents],
                'MultipleLines':[self.MultipleLines],
                'InternetService':[self.InternetService],
                'OnlineSecurity':[self.OnlineSecurity],
                'OnlineBackup':[self.OnlineBackup],
                'DeviceProtection':[self.DeviceProtection],
                'TechSupport':[self.TechSupport],
                'StreamingTV':[self.StreamingTV],
                'StreamingMovies':[self.StreamingMovies],
                'Contract':[self.Contract],
                'PaperlessBilling':[self.PaperlessBilling],
                'PaymentMethod':[self.PaymentMethod],
                'TenureMonths':[self.TenureMonths],
                'MonthlyCharges':[self.MonthlyCharges],
                'TotalCharges':[self.TotalCharges]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        
data = CustomData(
            SeniorCitizen='No',
            Partner='No',
            Dependents='yes',
            MultipleLines='No',
            InternetService='DSL',
            OnlineSecurity='yes',
            OnlineBackup='Yes',
            DeviceProtection='No',
            TechSupport='No',
            StreamingTV='No',
            StreamingMovies='No',
            Contract='Month-to-month',
            PaperlessBilling='No',
            PaymentMethod='Credit card (automatic)',
            TenureMonths=int(2),
            MonthlyCharges=float(2),
            TotalCharges=float(4)
        )

pred_df = data.get_data_as_data_frame()
print(pred_df)
predict_pipeline = PredictPipeline()
result = predict_pipeline.predict(pred_df)
print(result)
    