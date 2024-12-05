# Function to calculate the percentage of correct answers for each difficulty level
def calculate_percentage(total_questions, wrong_answers):
    correct_answers = total_questions - wrong_answers
    percentage = (correct_answers / total_questions) * 100
    return round(percentage, 2)

# Calculate the percentages for easy, medium, and hard levels
easy_percentage = calculate_percentage(student_answer_list['easy']['total_question'], student_answer_list['easy']['wrong'])
medium_percentage = calculate_percentage(student_answer_list['medium']['total_question'], student_answer_list['medium']['wrong'])
hard_percentage = calculate_percentage(student_answer_list['hard']['total_question'], student_answer_list['hard']['wrong'])

# Function to get suggestions based on percentage and difficulty
def get_suggestions(stream, difficulty, percentage, suggestion_list):
    for course in suggestion_list["courseList"]:
        if course["Stream"] == stream and course["Difficulty"] == difficulty:
            course_percentage = int(course["Percentage"].replace("%", ""))
            if course_percentage >= percentage:
                return course["Suggestion"]
    return None

# Get the suggestions for each difficulty level
easy_suggestions = get_suggestions(student_answer_list['Stream'], 'Easy', easy_percentage, suggestion_list)
medium_suggestions = get_suggestions(student_answer_list['Stream'], 'Medium', medium_percentage, suggestion_list)
hard_suggestions = get_suggestions(student_answer_list['Stream'], 'Hard', hard_percentage, suggestion_list)

# Output the results
print(f"Easy Question Percentage: {easy_percentage}%")
print("Easy Suggestions:")
if easy_suggestions:
    for suggestion in easy_suggestions:
        print(f"  Course: {suggestion['CourseName']}, Link: {suggestion['Course Link']}, Video: {suggestion['Course Video Link']}")
else:
    print("  No suggestions found.")

print(f"\nMedium Question Percentage: {medium_percentage}%")
print("Medium Suggestions:")
if medium_suggestions:
    for suggestion in medium_suggestions:
        print(f"  Course: {suggestion['CourseName']}, Link: {suggestion['Course Link']}, Video: {suggestion['Course Video Link']}")
else:
    print("  No suggestions found.")

print(f"\nHard Question Percentage: {hard_percentage}%")
print("Hard Suggestions:")
if hard_suggestions:
    for suggestion in hard_suggestions:
        print(f"  Course: {suggestion['CourseName']}, Link: {suggestion['Course Link']}, Video: {suggestion['Course Video Link']}")
else:
    print("  No suggestions found.")