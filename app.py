from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

#route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            SeniorCitizen=request.form.get('SeniorCitizen'),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            MultipleLines=request.form.get('MultipleLines'),
            InternetService=request.form.get('InternetService'),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            StreamingTV=request.form.get('StreamTV'),
            StreamingMovies=request.form.get('StreamingMovies'),
            Contract=request.form.get('Contract'),
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=request.form.get('PaymentMethod'),
            TenureMonths=int(request.form.get('TenureMonths')),
            MonthlyCharges=float(request.form.get('MonthlyCharges')),
            TotalCharges=float(request.form.get('TotalCharges'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print('Before Prediction')

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        print('After Prediction')
        return render_template('home.html', results=result[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)