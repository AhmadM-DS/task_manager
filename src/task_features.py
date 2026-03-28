"""
task_features.py

Task features that will be used in the logistic regression model.
"""

deadline_keywords = ["due", "tomorrow", "tonight", "deadline", "urgent", "asap", "today"]

difficulty_keywords = {
    "hard": ["research", "study", "analyze", "write", "develop", "solve", "review"],
    "medium": ["read", "prepare", "organize", "plan", "update"],
    "easy": ["email", "call", "buy", "text", "check", "print", "submit"]
}

category_keywords = {
    "academic": ["homework", "study", "exam", "quiz", "assignment", "essay", "lecture"],
    "health": ["workout", "exercise", "gym", "sleep", "eat", "meditate"],
    "social": ["call", "text", "email", "meet", "friend", "family"],
    "personal": ["buy", "clean", "organize", "pay", "schedule", "apply"]
}

