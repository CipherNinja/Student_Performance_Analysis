import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
# Update 'path_to_your_dataset' with the actual path of your dataset
df = pd.read_csv(r'C:\Users\Priyesh Pandey\OneDrive\Desktop\student performance\Dataset\Student_Performance.csv')

# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Extracurricular Activities'] = label_encoder.fit_transform(df['Extracurricular Activities'])

# Optional: Scale the features
scaler = StandardScaler()
df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']] = scaler.fit_transform(
    df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']]
)

# Create feature and target datasets
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed data for model training
# Update 'path_to_your_preprocessed_data' with the desired path for saving the preprocessed data
df.to_csv(r'C:\Users\Priyesh Pandey\OneDrive\Desktop\student performance\PreprocessedData\preprocessed_student_performance.csv', index=False)

print("Preprocessing complete. Preprocessed data saved to 'path_to_your_preprocessed_data'.")
