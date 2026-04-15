"""
combine_features.py

Combines personality dimension scores and task features for logistic regression model.
"""

from src.constants import CATEGORY_MAP

#Combining the features into one dataset
def combine_features(averages, tasks):
    combined = []

    for task in tasks:
        row = {
            "name": task["name"],
            "U": round(averages["U"], 2),
            "I": round(averages["I"], 2),
            "Q": round(averages["Q"], 2),
            "S": round(averages["S"], 2),
            "category": CATEGORY_MAP[task["category"]],
            "hours_per_week": task["hours_per_week"],
            "stress": task["stress"],
            "urgency": task["urgency"],
            "importance": task["importance"],
            "mental_effort": task["mental_effort"]
        }
        combined.append(row)

    return combined