"""
combine_features.py

Combines personality dimension scores and task features for logistic regression model.
"""

category_map = {
    "School Work": 0,
    "Physical Activity": 1,
    "Hobbies": 2,
    "Social Activities": 3,
    "Errands": 4,
    "Leisure / Down Time": 5,
    "Health / Grooming": 6,
    "Miscellaneous Projects": 7,
    "Work": 8
}

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
            "category": category_map[task["category"]],
            "hours_per_week": task["hours_per_week"],
            "stress": task["stress"],
            "urgency": task["urgency"],
            "importance": task["importance"],
            "mental_effort": task["mental_effort"]
        }
        combined.append(row)

    return combined