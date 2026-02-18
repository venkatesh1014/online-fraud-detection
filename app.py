from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load both model and scaler
with open('payments.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    scaler = saved_data['scaler']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        input_features = [float(x) for x in request.form.values()]
        features_array = np.array([input_features])
        
        # 1. Scale the input using the saved scaler
        scaled_features = scaler.transform(features_array)
        
        # 2. Predict
        prediction = model.predict(scaled_features)
        
        result = "Fraudulent" if prediction[0] == 1 else "Legitimate (Not Fraud)"
        
        return render_template('submit.html', prediction_text=f'Result: This transaction is {result}')

if __name__ == '__main__':
    app.run(debug=True)