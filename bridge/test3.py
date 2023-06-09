import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tkinter import *

# Load the dataset from a CSV file
data = pd.read_csv('bridge_accelerometer_data.csv')

# Separate the features (time, ax, ay, az) and the target variable (aT - strength)
X = data[['time', 'ax', 'ay', 'az']]
y = data['aT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate multiple machine learning algorithms
algorithms = {
    'Random Forest': RandomForestRegressor(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Linear Regression': LinearRegression(),
    'Support Vector Machine': SVR()
}

results = {}

for name, regressor in algorithms.items():
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = regressor.score(X_test, y_test)
    results[name] = accuracy

# Determine the strength of the bridge based on the best performing algorithm
best_algorithm = max(results, key=results.get)
bridge_strength = results[best_algorithm]

# Create a Tkinter user interface for visualizing the analysis results
root = Tk()

def plot_bar_graph():
    plt.bar(results.keys(), results.values())
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of ML Algorithms')
    plt.show()

def plot_line_graph():
    plt.plot(data['time'], data['aT'])
    plt.xlabel('Time')
    plt.ylabel('Strength')
    plt.title('Bridge Strength Over Time')
    plt.show()

bar_graph_button = Button(root, text="Plot Bar Graph", command=plot_bar_graph)
bar_graph_button.pack()

line_graph_button = Button(root, text="Plot Line Graph", command=plot_line_graph)
line_graph_button.pack()

root.mainloop()

print(f"The strength of the bridge, determined by the best performing algorithm ({best_algorithm}), is {bridge_strength}.")
