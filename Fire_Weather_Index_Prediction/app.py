import pickle
import bz2
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app_logger import log
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

# Import Classification and Regression model file
R_pickle_in = bz2.BZ2File('model/regression.pkl', 'rb')
model_R = pickle.load(R_pickle_in)


# Route for homepage
@app.route('/')
def home():
    log.info('Home page loaded successfully')
    return render_template('index.html')


# Route for API Testing
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        log.info('Input from Api testing', data)
        new_data = [list(data.values())]
        final_data = scaler.transform(new_data)
        output = model_R.predict(data)[0]
        return jsonify(f"Temprature will be: {output}")
    except Exception as e:
        output = 'Check the in input again!'
        log.error('error in input from Postman', e)
        return jsonify(output)


# Run APP in Debug mode
if __name__ == "__main__":
    app.run(debug=False)
