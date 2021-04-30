# FLask API is a tool which helps to connect webs servers to your project

from flask import Flask, render_template, url_for, request, jsonify
import joblib
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('home.html')

@app.route('/result',methods=['POST','GET'])
def result():
    cylinders=request.form["cylinders"]
    displacement=request.form["displacement"]
    horsepower=request.form["horsepower"]
    weight=request.form["weight"]
    acceleration=request.form["acceleration"]
    model_year=request.form["model_year"]
    origin=request.form["origin"]
    
    values=[[cylinders,displacement,horsepower,weight,acceleration,model_year,origin]]

    scaler_path=os.path.join(os.path.dirname('D:/PROJECTS/Fuel-Efficiency-Prediction-using-ANN-main/'), 'scaler3.pkl')

    sc=None
    with open(scaler_path, 'rb') as f:
        sc = pickle.load(f)
        
    values = sc.transform(values)

    model = load_model(r"D:\PROJECTS\Fuel-Efficiency-Prediction-using-ANN-main\models\model.h5")

    prediction = model.predict(values)
    prediction = float(prediction)
    output = round(prediction, 2)
    
    return render_template('home.html', prediction_text=" Miles per Gallon [MPG] : {}".format(output))
    
if __name__=="__main__":
    app.run(debug=True,port=3298)