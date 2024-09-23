# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:40:26 2024

@author: lenovo
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

# Load the data (replace 'PredictionsData.xlsx' with the actual path to your dataset)
data = pd.read_excel('PredictionsData.xlsx')

# Standardize the features
scaler = StandardScaler()
X = data.drop('Grade', axis=1)
y = data['Grade']
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network architecture
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Display the model's architecture
model.summary()

# Train the neural network model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test data
loss, mse = model.evaluate(X_test, y_test, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


# Hyperparameter Tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV

try:
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
except ImportError as e:
    print(f"Error importing KerasRegressor: {e}")
    raise

def create_model(optimizer='adam', dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the grid search parameters
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [32, 64],
    'epochs': [50, 100],
    'dropout_rate': [0.3, 0.5, 0.7]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Summarize results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Get the best model
best_model = grid_result.best_estimator_

# Evaluate the model on the test data
mse, _ = best_model.model.evaluate(X_test, y_test, verbose=1)
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')