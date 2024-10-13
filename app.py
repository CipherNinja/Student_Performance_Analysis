from flask import Flask, request, jsonify, render_template
from Machine_Code.testmodel import get_student_performance_json  # Import the function from test_model.py
import json  # <-- Add this import

# Initialize Flask app
app = Flask(__name__)

# Serve the index.html page
@app.route('/')
def home():
    return render_template('index.html')

# Define API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON from the form
    input_data = request.get_json()[0]

    # Extract individual features
    hours_studied = input_data['Hours Studied']
    sleep_hours = input_data['Sleep Hours']
    extracurricular_activities = input_data['Extracurricular Activities']
    sample_papers_practiced = input_data['Sample Question Papers Practiced']
    previous_scores = input_data['Previous Scores']

    # Call the function from test_model.py to get the prediction and analysis in JSON
    result_json = get_student_performance_json(hours_studied, sleep_hours, extracurricular_activities, sample_papers_practiced, previous_scores)
    print(result_json)
    # Convert the JSON string to Python dict
    result_data = json.loads(result_json)

    # Return the entire analysis as JSON response (not just the prediction)
    return jsonify(result_data)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
