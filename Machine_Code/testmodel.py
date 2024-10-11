import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
model = load_model(r'C:\Users\Priyesh Pandey\OneDrive\Desktop\Student_Performance_Analysis\model\student_performance_model.h5')

# Preprocess functions for scaling and encoding
scaler = StandardScaler()
label_encoder = LabelEncoder()
label_encoder.fit(['Yes', 'No'])  # Assuming this was pre-fitted on training data

# Test data (Good and Bad performance parameters)
test_data = pd.DataFrame({
    'Hours Studied': [8, 2, 7, 1, 6, 3, 9, 1, 5, 2],  # Good: 7-9, Bad: 1-3
    'Previous Scores': [90, 45, 85, 50, 88, 52, 99, 40, 75, 48],  # Good: 85-99, Bad: 40-52
    'Extracurricular Activities': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes'],  # Mixed
    'Sleep Hours': [8, 4, 7, 5, 9, 6, 8, 4, 7, 5],  # Good: 7-9, Bad: <6 or excessive
    'Sample Question Papers Practiced': [7, 1, 9, 2, 8, 3, 9, 1, 7, 2]  # Good: 7-9, Bad: 1-3
})

# Encode categorical variables
test_data['Extracurricular Activities'] = label_encoder.transform(test_data['Extracurricular Activities'])

# Save original data before scaling for displaying purposes
original_test_data = test_data.copy()

# Scale the numeric features
test_data[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']] = scaler.fit_transform(
    test_data[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']]
)

# Make predictions
predictions = model.predict(test_data)

# Ensure predictions are within the 10-100 range
predictions = np.clip(predictions, 10, 100)  # Clip values between 10 and 100

# Add predictions to the original (non-scaled) data for display
original_test_data['Predicted Performance'] = predictions

# Display the full test cases with predictions
print(original_test_data.to_string(index=False))
