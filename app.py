import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)  # Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))


app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # spread request form

    body = request.form
    # get gender from body
    gender = 1 if body.get("gender") == 1 else 0
    SeniorCitizen = 1 if body.get("SeniorCitizen") == 1 else 0
    Partner = 1 if body.get("Partner") == "1" else 0
    Dependents = 1 if body.get("Dependents") == "1" else 0
    tenure = int(body.get("tenure"))
    PhoneService = 1 if body.get("PhoneService") == "1" else 0
    MultipleLines = 1 if body.get("MultipleLines") == "1" else 0
    OnlineSecurity = 1 if body.get("OnlineSecurity") == "1" else 0
    OnlineBackup = 1 if body.get("OnlineBackup") == "1" else 0
    DeviceProtection = 1 if body.get("DeviceProtection") == "1" else 0
    TechSupport = 1 if body.get("TechSupport") == "1" else 0
    StreamingTV = 1 if body.get("StreamingTV") == "1" else 0
    StreamingMovies = 1 if body.get("StreamingMovies") == "1" else 0
    Contract = int(body.get("Contract"))
    PaperlessBilling = 1 if body.get("PaperlessBilling") == "1" else 0
    MonthlyCharges = int(body.get("MonthlyCharges"))
    TotalCharges = int(body.get("TotalCharges"))
    InternetService_DSL = 0
    InternetService_Fiber = 0
    InternetService_No = 0
    service = body.get("service")
    if(service == "dsl"):
        InternetService_DSL = 1
    elif(service == "fiber"):
        InternetService_Fiber = 1
    elif(service == "no"):
        InternetService_No = 1

    PaymentMethod_Bank = 0
    PaymentMethod_Credit = 0
    PaymentMethod_Electronic = 0
    PaymentMethod_Mailed = 0
    payment = body.get("payment")
    if(payment == "bank"):
        PaymentMethod_Bank = 1
    elif(payment == "credit"):
        PaymentMethod_Credit = 1
    elif(payment == "electronic"):
        PaymentMethod_Electronic = 1
    elif(payment == "mailed"):
        PaymentMethod_Mailed = 1

    values = [
        gender, SeniorCitizen, Partner, Dependents,
        tenure, PhoneService, MultipleLines,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        Contract, PaperlessBilling, MonthlyCharges, TotalCharges, InternetService_DSL, InternetService_Fiber,
        InternetService_No, PaymentMethod_Bank, PaymentMethod_Credit, PaymentMethod_Electronic, PaymentMethod_Mailed
    ]
    final_features = [np.array(values)]
    prediction = model.predict(final_features)

    if prediction == 0:
        output = 'Not Churn'
    else:
        output = 'Churn'
    response = jsonify({
        'prediction': int(prediction),
        'message': "Customer will be "+output})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    app.run(debug=True)
