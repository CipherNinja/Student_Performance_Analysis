import tensorflow as tf
import numpy as np
import json

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\Priyesh Pandey\OneDrive\Desktop\Student_Performance_Analysis\model\student_performance_model.h5')

# Custom logic for Balanced and Imbalanced Workload
def analyze_workload(hours_studied, sleep_hours, extracurricular_activities, sample_papers_practiced, previous_scores):
    # Analyze based on the logic of Balanced Workload
    if (5 <= hours_studied <= 8) and (7 <= sleep_hours <= 9) and (extracurricular_activities == 'Yes') and (sample_papers_practiced >= 3):
        workload = "Balanced Workload"
        performance_comment = "The student is likely to perform optimally."
    elif (sleep_hours < 6) or (hours_studied < 4) or (previous_scores < 60 and sample_papers_practiced > 5):
        workload = "Imbalanced Workload"
        performance_comment = "The student might be at risk of poor performance."
    else:
        workload = "Moderate Workload"
        performance_comment = "Workload seems moderate, performance may vary."
    
    # Return the analysis result
    return workload, performance_comment

# Function to get performance prediction and detailed analysis as JSON
def get_student_performance_json(hours_studied, sleep_hours, extracurricular_activities, sample_papers_practiced, previous_scores):
    # Encode extracurricular activities
    extracurricular_encoded = 1 if extracurricular_activities == 'Yes' else 0
    
    # Compute interaction features
    study_sleep_balance = hours_studied * sleep_hours
    practice_effectiveness = previous_scores * sample_papers_practiced
    extracurricular_impact = extracurricular_encoded * hours_studied
    
    # Prepare input with all required features (including interaction features)
    input_data = np.array([[hours_studied, previous_scores, extracurricular_encoded, sleep_hours, 
                            sample_papers_practiced, study_sleep_balance, practice_effectiveness, extracurricular_impact]])
    
    # Make prediction using the trained model
    predicted_performance = model.predict(input_data)
    
    # Get workload analysis
    workload, performance_comment = analyze_workload(hours_studied, sleep_hours, extracurricular_activities, sample_papers_practiced, previous_scores)
    
    # Prepare the JSON data
    student_data = {
        "input": {
            "Hours Studied": hours_studied,
            "Sleep Hours": sleep_hours,
            "Extracurricular Activities": extracurricular_activities,
            "Sample Question Papers Practiced": sample_papers_practiced,
            "Previous Scores": previous_scores
        },
        "prediction": {
            "Predicted Performance Index": round(predicted_performance[0][0], 2)
        },
        "analysis": {
            "Workload": workload,
            "Performance Comment": performance_comment
        }
    }
    
    # Convert to JSON string
    return json.dumps(student_data, indent=4)

# Example usage
json_result1 = get_student_performance_json(7, 8, 'Yes', 5, 75)
json_result2 = get_student_performance_json(4, 5, 'No', 6, 50)

# Print JSON results
print(json_result1)
print(json_result2)
