import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Load the saved model and final_df
pipe = pickle.load(open('model_pipeline.pkl', 'rb'))
final_df = pickle.load(open('final_df.pkl', 'rb'))

# Create a Flask app
app = Flask(__name__)

# Define the endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json
    
    # Extract data from the request
    batting_team = data['batting_team']
    bowling_team = data['bowling_team']
    city = data['city']
    target = data['target']
    score = data['score']
    wickets = data['wickets']
    overs = data['overs']

    # Calculate additional match data
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_remaining = 10 - wickets
    crr = score / overs if overs > 0 else 0  # Prevent division by zero
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # Prevent division by zero

    # Create input dataframe for prediction
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_remaining': [wickets_remaining],
        'total_run_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Make predictions using the loaded model
    result = pipe.predict_proba(input_df)

    # Extract win probabilities
    win_prob_batting_team = round(result[0][1] * 100)
    win_prob_bowling_team = round(result[0][0] * 100)

    # Return the results as JSON
    return jsonify({
        'batting_team': batting_team,
        'win_probability': win_prob_batting_team,
        'bowling_team': bowling_team,
        'win_probability_bowling_team': win_prob_bowling_team
    })

if __name__ == '__main__':
    app.run(debug=True)
