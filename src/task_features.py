"""
task_features.py

Task features that will be used in the logistic regression model.
"""

from src.constants import TASK_CATEGORIES

#Validates rating input
def get_valid_rating(prompt):
    while True:
        try:
            value = int(input(prompt))
            if 1 <= value <= 5:
                return value
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

#Validates numeric input
def get_valid_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value >= 0:
                return value
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_task_input():
    print("\nEnter your to-do list. Type 'done' when finished.\n")
    tasks = []

    while True:
        task_name = input("Task name (or 'done' to finish): ").strip()
        if task_name.lower() == "done":
            break

        print("\nSelect a category:")
        for i, category in enumerate(TASK_CATEGORIES):
            print(f"  {i + 1}. {category}")

        while True:
            try:
                cat_choice = int(input("Category (1-9): "))
                if 1 <= cat_choice <= 9:
                    category = TASK_CATEGORIES[cat_choice - 1]
                    break
                else:
                    print("Please enter a number between 1 and 9.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        hours_per_week = get_valid_float("Hours per week spent on this task: ")
        stress = get_valid_rating("Stress level (1-5): ")
        urgency = get_valid_rating("Urgency (1-5): ")
        importance = get_valid_rating("Importance (1-5): ")
        mental_effort = get_valid_rating("Mental effort required (1-5): ")

        tasks.append({
            "name": task_name,
            "category": category,
            "hours_per_week": hours_per_week,
            "stress": stress,
            "urgency": urgency,
            "importance": importance,
            "mental_effort": mental_effort
        })

        print(f"\n'{task_name}' added.\n")

    return tasks