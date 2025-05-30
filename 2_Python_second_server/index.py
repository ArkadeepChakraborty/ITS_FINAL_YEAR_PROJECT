from flask import Flask, jsonify, request
from flask_cors import CORS
from analys_suggestion import get_suggestions, model, le_difficulty, le_stream, le_course, scaler, df

app = Flask(__name__) 
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Course Suggestion API"

@app.route('/course_suggestion', methods=['POST'])
def get_course_suggestion(): 
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    user_level_sheet = data.get('user_level_sheet', {})
    
    # Get personalized course suggestions
    suggestions = get_suggestions(user_level_sheet, model, le_difficulty, le_stream, le_course, scaler, df)

    return jsonify({'suggestions': suggestions, 'user_level_sheet': user_level_sheet})

if __name__ == '__main__':
    app.run(debug=True)




