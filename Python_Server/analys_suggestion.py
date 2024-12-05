import json

student_answer_list = {}

def get_suggestions(student_answer_list, suggestion_list):
    # Extract scores
    easy_score = 0
    medium_score = 0
    hard_score = 0
    stream = student_answer_list.get('Stream', None)  # Extract stream

    if student_answer_list['easy']['total_question'] != 0:
        easy_score = (
            (student_answer_list['easy']['total_question'] - student_answer_list['easy']['right']) /
            student_answer_list['easy']['total_question']
        ) * 100
    if student_answer_list['medium']['total_question'] != 0:
        medium_score = (
            (student_answer_list['medium']['total_question'] - student_answer_list['medium']['right']) /
            student_answer_list['medium']['total_question']
        ) * 100
    if student_answer_list['hard']['total_question'] != 0:
        hard_score = (
            (student_answer_list['hard']['total_question'] - student_answer_list['hard']['right']) /
            student_answer_list['hard']['total_question']
        ) * 100

    suggestions = []

    # Match suggestions by difficulty, percentage, and stream
    for course in suggestion_list["courseList"]:
        # Ensure the stream matches
        if course["Stream"] != stream:
            continue

        # Filter by difficulty and percentage
        if course["Difficulty"] == "Easy" and easy_score >= int(course["Percentage"].strip('%')):
            suggestions.append({"Difficulty": "Easy", "Courses": course["Suggestion"]})
        elif course["Difficulty"] == "Medium" and medium_score >= int(course["Percentage"].strip('%')):
            suggestions.append({"Difficulty": "Medium", "Courses": course["Suggestion"]})
        elif course["Difficulty"] == "Hard" and hard_score >= int(course["Percentage"].strip('%')):
            suggestions.append({"Difficulty": "Hard", "Courses": course["Suggestion"]})

    return suggestions


def get_suggestion_list(student_answer_list):
    # Validate input and extract the stream
    student_answer_list['Stream'] = student_answer_list.get('Stream', None)

    # Load suggestion list from a JSON file
    with open('suggestion_list.json', 'r') as file:
        suggestion_list = json.load(file)

    # Get suggestions for the student
    suggestions = get_suggestions(student_answer_list, suggestion_list)
    return suggestions


# Example usage
# student_answer_list = {
#     "Stream": "IT",
#     "easy": {"total_question": 10, "right": 6},
#     "medium": {"total_question": 8, "right": 5},
#     "hard": {"total_question": 5, "right": 2},
# }

# result = get_suggestion_list(student_answer_list)

# # Print suggestions
# for suggestion in result:
#     print(f"Difficulty: {suggestion['Difficulty']}")
#     for course in suggestion["Courses"]:
#         print(f"  - Course Name: {course['CourseName']}")
#         print(f"    Course Link: {course['CourseLink']}")
#         print(f"    Course Video Link: {course['CourseVideoLink']}")
#     print()
