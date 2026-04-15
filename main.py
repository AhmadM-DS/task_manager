from src.questionnaire import run_questionnaire, compute_profile
from src.task_features import get_task_input
from src.combine_features import combine_features
from src.prepare_data import load_and_prepare
from src.model import train_model, predict_tasks

#Step 1: Lifestyle questionnaire
"""scores = run_questionnaire()
averages, category = compute_profile(scores)"""
#for testing
averages = {"U": 3.0, "I": 3.0, "Q": 3.0, "S": 3.0}
category = "Planner"
"""print(f"\nYour dimension averages: {averages}")
print(f"Your personality category: {category}")"""

#Step 2: To-Do list input
"""tasks = get_task_input()"""
# TEMPORARY: Skip task input for testing
tasks = [
    {"name": "study for exam", "category": "School Work", "hours_per_week": 8.0, "stress": 5, "urgency": 5, "importance": 5, "mental_effort": 5},
    {"name": "do laundry", "category": "Errands", "hours_per_week": 1.0, "stress": 1, "urgency": 2, "importance": 2, "mental_effort": 1},
    {"name": "go to gym", "category": "Physical Activity", "hours_per_week": 3.0, "stress": 2, "urgency": 1, "importance": 4, "mental_effort": 2},
    {"name": "play video games", "category": "Leisure / Down Time", "hours_per_week": 10.0, "stress": 1, "urgency": 1, "importance": 1, "mental_effort": 1},
    {"name": "read textbook", "category": "School Work", "hours_per_week": 5.0, "stress": 3, "urgency": 4, "importance": 4, "mental_effort": 3},
    {"name": "call mom", "category": "Social Activities", "hours_per_week": 1.0, "stress": 1, "urgency": 3, "importance": 3, "mental_effort": 1},
    {"name": "work on resume", "category": "Miscellaneous Projects", "hours_per_week": 2.0, "stress": 2, "urgency": 3, "importance": 5, "mental_effort": 3},
    {"name": "grocery shopping", "category": "Errands", "hours_per_week": 1.5, "stress": 1, "urgency": 4, "importance": 3, "mental_effort": 1},
    {"name": "meditate", "category": "Health / Grooming", "hours_per_week": 0.5, "stress": 1, "urgency": 1, "importance": 3, "mental_effort": 1},
    {"name": "work shift", "category": "Work", "hours_per_week": 20.0, "stress": 3, "urgency": 5, "importance": 5, "mental_effort": 3}
]

# REAL: Uncomment when done testing
# tasks = get_task_input()
"""print("\n---Task list---")
for task in tasks:
    print(task)"""

#Step 3: Combine features
combined = combine_features(averages, tasks)
"""print("\n---Combined Features---")
for row in combined:
    print(row)"""

#Step 4: Load dataset & Train model
df = load_and_prepare("data/daily_activity_survey_data.xlsx")
model, scaler = train_model(df)

#Step 5: Predict and present
results = predict_tasks(model, scaler, combined)
print("\n--- Your Optimized To-Do List ---")
for i, task in enumerate(results):
    print(f"{i + 1}. {task['name']} — {task['completion_probability']:.2%}")