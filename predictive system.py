import pickle
import numpy as np


with open('fraud_detection_model.pkl', 'rb') as f:
    saved_objects = pickle.load(f)

model = saved_objects['model']
scaler = saved_objects['scaler']

# New data for prediction
new_data = np.array([[0.0, 0.5, 200.0, 1.0, 60, 50, 44, 11, 23, 5, 14, 7, 8, 4, 3, 2,
                      1.5, 6, 12, 0.4, 9, 11, 15, 10, 2, 8, 4, 0.9, 100, 0.1]])  # Example data
new_data_scaled = scaler.transform(new_data)

# Make predictions
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)[:, 1]

print(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
print(f"Probability of Fraud: {probability[0]:.2f}")
