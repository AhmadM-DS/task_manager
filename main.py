"""
main.py

Entry point for the Automated Task Management System.
Orchestrates the full pipeline:
    1. Collect user lifestyle data via questionnaire
    2. Collect user to-do list
    3. Combine features into model-ready format
    4. Load or train logistic regression model
    5. Predict task completion probabilities
    6. Present optimized to-do list sorted by probability
"""

import os
from src.questionnaire import run_questionnaire, compute_profile
from src.task_features import get_task_input
from src.combine_features import combine_features
from src.prepare_data import load_and_prepare
from src.model import train_model, predict_tasks, save_model, load_model

#Step 1: Lifestyle questionnaire
scores = run_questionnaire()
averages, category = compute_profile(scores)

print(f"\nYour dimension averages: {averages}")
print(f"Your personality category: {category}")

#Step 2: To-Do list input
tasks = get_task_input()

#Step 3: Combine features
combined = combine_features(averages, tasks)

#Step 4: Load dataset & Train model
if os.path.exists("models/model.pkl"):
    model, scaler = load_model()
else:
    df = load_and_prepare("data/daily_activity_survey_data.xlsx")
    model, scaler = train_model(df)
    save_model(model, scaler)

#Step 5: Predict and present
results = predict_tasks(model, scaler, combined)
print("\n--- Your Optimized To-Do List ---")
for i, task in enumerate(results):
    print(f"{i + 1}. {task['name']} — {task['completion_probability']:.2%}")