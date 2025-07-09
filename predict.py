# predict.py
import joblib

# Load model
model = joblib.load("iris_model.pkl")

# Sample input (4 features: sepal length, sepal width, petal length, petal width)
sample = [[5.1, 3.5, 1.4, 0.2]]  # Change values to test different predictions

# Predict
prediction = model.predict(sample)

# Show result
species = ['setosa', 'versicolor', 'virginica']
print("Predicted species:", species[prediction[0]])
