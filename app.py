from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import this
import joblib

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

model = joblib.load("iris_model.pkl")
species = ['setosa', 'versicolor', 'virginica']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]
    prediction = model.predict([features])
    return jsonify({'prediction': species[prediction[0]]})

if __name__ == '__main__':
    app.run(debug=True)
