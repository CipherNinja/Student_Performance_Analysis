import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import preprocess_training_data
import numpy as np

# Load and preprocess the dataset for training
X_train, y_train = preprocess_training_data(r'C:\Users\Priyesh Pandey\OneDrive\Desktop\Student_Performance_Analysis\Dataset\Student_Performance.csv')

# Split the data for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the neural network model with sigmoid activation in the output layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Scale the predictions to fit the range of 10 to 100
y_pred_scaled = 10 + (y_pred * 90)  # Since sigmoid output is between 0 and 1, rescale to 10-100 range

# Evaluate the model performance with scaled predictions
mse = mean_squared_error(y_test, y_pred_scaled)
r2 = r2_score(y_test, y_pred_scaled)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the trained model
model.save(r'C:\Users\Priyesh Pandey\OneDrive\Desktop\Student_Performance_Analysis\model\student_performance_model.h5')
print("Model saved at 'C:\\Users\\Priyesh Pandey\\OneDrive\\Desktop\\student_performance\\model\\student_performance_model.h5'")
