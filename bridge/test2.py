import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load the dataset
data = pd.DataFrame({
    'time': [0.005, 0.023, 0.04, 0.059, 0.078, 0.097, 0.117],
    'ax': [-0.0112, -0.0721, 0.0073, -0.127, 0.0337, -0.0404, 0.0154],
    'ay': [0.0159, -0.0435, -0.0086, 0.0126, 0.0125, -0.0186, -0.0503],
    'az': [0.0539, -0.1818, -0.1048, -0.1444, 0.1407, -0.1674, -0.1025],
    'aT': [0.057, 0.2, 0.105, 0.193, 0.145, 0.173, 0.115],
    'condition': ['Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad', 'Good']
})

# Separate the features (sensor readings) and the target variable (bridge condition)
X = data.drop('condition', axis=1)
y = data['condition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate multiple machine learning algorithms
algorithms = {
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC()
}

for name, classifier in algorithms.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")

# Predict the condition of a new bridge
new_bridge_data = pd.DataFrame({
    'time': [0.1],
    'ax': [0.025],
    'ay': [-0.016],
    'az': [-0.073],
    'aT': [0.09]
})

predicted_condition = algorithms['Random Forest'].predict(new_bridge_data)
print(f"Predicted Condition: {predicted_condition}")

# Plot the bridge strength
plt.plot(data['time'], data['aT'])
plt.xlabel('Time')
plt.ylabel('Strength')
plt.title('Bridge Strength Over Time')
plt.show()
