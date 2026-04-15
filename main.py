from src.questionnaire import run_questionnaire, compute_profile
from src.task_features import get_task_input
from src.combine_features import combine_features

#Step 1: Lifestyle questionnaire
scores = run_questionnaire()
averages, category = compute_profile(scores)

#Step 2: To-Do list input
tasks = get_task_input()

#Step 3: Combine features
combined = combine_features(averages, tasks)

#Displaying output at once
print(f"Your dimension averages: {averages}")
print(f"Your personality category: {category}")
print("\n---Task list---")
for task in tasks:
    print(task)
print("\n---Combined Features---")
for row in combined:
    print(row)