from flask import Flask, render_template, request
from joblib import load
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
#from tensorflow import keras
from keras.models import load_model


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    request_type = request.method
    if request_type == 'GET':
        return render_template('index.html', prediction='N/A')
    else:
        # Taking user inputs and transforming them into a numpy array and scaling them
        user_inputs = request.form
        scaled_data_array = transform_user_inputs(user_inputs)

        # Loading the model and making a prediction
        model = load_model('app/model.h5')
        prediction = model.predict(scaled_data_array)
        
        # As our model aswers the question "What is the probability of a loan NOT being paid off?",
        # we need to reverse the prediction for it to be more intuitive for a user
        reversed_prediction = round(1 - prediction[0][0], 2)
        # Bringing the prediction to a percentage
        reversed_prediction_pct = reversed_prediction * 100
        
        # This value is used on html page to set the color of the prediction box
        # the higher the value the greener the box
        prediction_normalized = (reversed_prediction - 0.5) * 2


        return render_template('index.html', prediction=reversed_prediction_pct, prediction_normalized=prediction_normalized)
    
    
def transform_user_inputs(user_inputs):
    
    # Here we are transforming the user inputs into proper format for our model
    int_rate = float(user_inputs['int_rate']) / 100
    installment = float(user_inputs['installment'])
    log_annual_inc = math.log(float(user_inputs['annual_inc']))
    dti = float(user_inputs['dti'])
    fico = float(user_inputs['fico'])
    days_with_cr_line = float(user_inputs['days_with_cr_line'])
    revol_bal = float(user_inputs['revol_bal'])
    revol_util = float(user_inputs['revol_util'])
    inq_last_6mths = float(user_inputs['inq_last_6mths'])
    delinq_2yrs = float(user_inputs['delinq_2yrs'])
    pub_rec = float(user_inputs['pub_rec'])
    purpose_array = np.eye(7)[['all_other', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'major_purchase', 'small_business'].index(user_inputs['purpose'])]
    dti_pct = dti * 100 / log_annual_inc
    credit_utilization = revol_bal * revol_util

    # We are creating a numpy arrays with all the inputs, transformed inputs and auxilarry data
    user_inputs_array = np.array([int_rate, installment, log_annual_inc, dti, fico, days_with_cr_line, revol_bal, revol_util, inq_last_6mths, delinq_2yrs, pub_rec])
    dti_pct = np.array([dti_pct])
    credit_utilization = np.array([credit_utilization])

    # Concatenating all the arrays into one, in the proper order for our model
    data_array = np.concatenate([user_inputs_array, purpose_array, dti_pct, credit_utilization]).reshape(1, 20)
    
    # We are also scaling the data using the scaler we saved in the training step
    scaler = load('app/scaler.joblib')
    scaled_data_array = scaler.transform(data_array)
    
    return scaled_data_array