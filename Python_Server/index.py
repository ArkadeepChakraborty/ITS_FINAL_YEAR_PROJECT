from flask import Flask, jsonify, request  # Import request from Flask
from flask_cors import CORS  # Import CORS for handling cross-origin requests
import json

from analys_suggestion import get_suggestion_list


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def index():
    return "Welcome in REST API using Python"


@app.route('/course_suggestion', methods=['POST'])
def get_course_suggestion():
    # Use request.get_json() to extract JSON data from the POST request
    data = request.get_json()  # This parses the JSON sent in the request body
    if not data:
        return jsonify({'error': 'No data provided'}), 400  # Return a 400 error if no data is provided

    # Extract 'user_level_sheet' from the request data
    user_level_sheet = data.get('user_level_sheet', {})
    
    #print(user_level_sheet)

    # In your example, it seems you're trying to access 'right_answer' from user_level_sheet
    # Modify this as needed depending on your data structure
    suggestion = get_suggestion_list(user_level_sheet)
    
    # Return the suggestion in the response
    return jsonify({'suggestion': suggestion, 'user_level_sheet': user_level_sheet})

if __name__ == '__main__':
    app.run(debug=True)