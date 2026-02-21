from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('C:\Desktop\House_prediction_model\model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['area'],
        data['bedrooms'],
        data['bathrooms'],
        data['age']
    ]])
    prediction = model.predict(features)[0]
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)