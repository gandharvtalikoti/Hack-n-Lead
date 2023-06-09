import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

# Load the dataset from CSV file
data = pd.read_csv('bridge_accelerometer_data.csv')

# Separate the features (time, ax, ay, az) and the target variable (aT - strength of the bridge)
X = data[['time', 'ax', 'ay', 'az']]
y = data['aT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using Min-Max scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate multiple machine learning algorithms
algorithms = {
    'Random Forest': RandomForestRegressor(),
    'Linear Regression': LinearRegression(),
    'Support Vector Machine': SVR()
}

for name, model in algorithms.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = model.score(X_test_scaled, y_test)
    print(f"{name} R2 Score: {accuracy}")

# Predict the strength of a new bridge
new_bridge_data = pd.DataFrame({
    'time': [0.2],
    'ax': [0.012],
    'ay': [-0.021],
    'az': [0.035]
})

new_bridge_data_scaled = scaler.transform(new_bridge_data)
predicted_strength = algorithms['Random Forest'].predict(new_bridge_data_scaled)
print(f"Predicted Strength: {predicted_strength}")

# Plot the strength of the bridge over time
plt.plot(data['time'], data['aT'])
plt.xlabel('Time')
plt.ylabel('Strength')
plt.title('Bridge Strength Over Time')
plt.show()
