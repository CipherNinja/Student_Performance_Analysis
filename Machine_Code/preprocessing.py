import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Initialize encoders and scalers
label_encoder = LabelEncoder()
scaler = StandardScaler()

def fit_encoders_and_scalers(df):
    global label_encoder, scaler
    label_encoder.fit(df['Extracurricular Activities'])

    # Define the columns to scale (including the newly created interaction features)
    features_to_scale = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 
                         'Sample Question Papers Practiced', 'Study_Sleep_Balance', 
                         'Practice_Effectiveness', 'Extracurricular_Impact']

    # Fit the scaler on these features
    scaler.fit(df[features_to_scale])

def preprocess_training_data(path_to_csv):
    # Load the dataset
    df = pd.read_csv(path_to_csv)

    # Handle missing values: only fill missing values for numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Encode categorical variables
    df['Extracurricular Activities'] = label_encoder.fit_transform(df['Extracurricular Activities'])

    # Create new interaction features before fitting the scaler
    df['Study_Sleep_Balance'] = df['Hours Studied'] * df['Sleep Hours']
    df['Practice_Effectiveness'] = df['Previous Scores'] * df['Sample Question Papers Practiced']
    df['Extracurricular_Impact'] = df['Extracurricular Activities'] * df['Hours Studied']

    # Now that the interaction features are created, fit the encoders and scalers
    fit_encoders_and_scalers(df)

    # Scale the features
    features_to_scale = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 
                         'Sample Question Papers Practiced', 'Study_Sleep_Balance', 
                         'Practice_Effectiveness', 'Extracurricular_Impact']

    df[features_to_scale] = scaler.transform(df[features_to_scale])

    # Split features and target
    X = df.drop('Performance Index', axis=1)
    y = df['Performance Index']

    # Save preprocessed data for future reference
    df.to_csv(r'C:\Users\Priyesh Pandey\OneDrive\Desktop\Student_Performance_Analysis\PreprocessedData\preprocessed_student_performance.csv', index=False)
    print("Preprocessed training data saved to PreprocessedData folder.")

    # Save the label encoder and scaler
    joblib.dump(label_encoder, r'C:\Users\Priyesh Pandey\OneDrive\Desktop\Student_Performance_Analysis\model\label_encoder.pkl')
    joblib.dump(scaler, r'C:\Users\Priyesh Pandey\OneDrive\Desktop\Student_Performance_Analysis\model\scaler.pkl')
    print("Label encoder and scaler saved to model folder.")
    
    return X, y

# If this script is executed directly, preprocess the training data
if __name__ == "__main__":
    preprocess_training_data(r'C:\Users\Priyesh Pandey\OneDrive\Desktop\Student_Performance_Analysis\Dataset\Student_Performance.csv')
