from src.questionnaire import run_questionnaire, compute_profile
from src.task_features import get_task_input
from src.combine_features import combine_features
from src.prepare_data import load_and_prepare
from src.model import train_model

#Step 1: Lifestyle questionnaire
scores = run_questionnaire()
averages, category = compute_profile(scores)

#Step 2: To-Do list input
tasks = get_task_input()

#Step 3: Combine features
combined = combine_features(averages, tasks)

#Step 4: Load dataset & Train model
df = load_and_prepare("data/daily_activity_survey_data.xlsx")
model, scaler = train_model(df)

#Displaying output at once
print(f"\nYour dimension averages: {averages}")
print(f"Your personality category: {category}")
print("\n---Task list---")
for task in tasks:
    print(task)
print("\n---Combined Features---")
for row in combined:
    print(row)