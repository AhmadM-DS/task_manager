"""
app.py

Streamlit UI for the Automated Task Management System.
Provides a web interface for the questionnaire, task input, and results.
"""

import os
import streamlit as st
from src.questionnaire import questions, compute_profile
from src.combine_features import combine_features
from src.prepare_data import load_and_prepare
from src.model import train_model, predict_tasks, save_model, load_model
from src.constants import TASK_CATEGORIES

st.set_page_config(page_title="Task Manager", layout="centered")

st.title("Automated Task Manager")
st.markdown("Answer the questionnaire below to get a personalized task schedule.")

#Step 1: Questionnaire
st.header("Step 1: Lifestyle Questionnaire")

with st.form("questionnaire_form"):
    answers = {}
    for i, question in enumerate(questions):
        options = [option[0] for option in question["options"]]
        answer = st.radio(
            label=f"**Q{i + 1}. {question['text']}**",
            options=options,
            index=None,
            key=f"q{i}"
        )
        answers[i] = (answer, question)

    submitted = st.form_submit_button("Submit Questionnaire")

if submitted:
    if any(answer is None for answer, _ in answers.values()):
        st.warning("Please answer all questions before submitting.")
    else:
        scores = {"U": [], "I": [], "Q": [], "S": []}
        for answer, question in answers.values():
            selected_score = dict(question["options"])[answer]
            scores[question["dimension"]].append(selected_score)
        averages, category = compute_profile(scores)
        st.session_state["averages"] = averages
        st.session_state["category"] = category

if "category" in st.session_state:
    st.success(f"Your personality type: **{st.session_state['category']}**")

#Step 2: Task Input
if "averages" in st.session_state:
    st.header("Step 2: Enter Your To-Do List")

    if "tasks" not in st.session_state:
        st.session_state["tasks"] = []

    if "last_added" not in st.session_state:
        st.session_state["last_added"] = None
    
    if "form_counter" not in st.session_state:
        st.session_state["form_counter"] = 0

    form_key = f"task_form_{st.session_state['form_counter']}"

    with st.form(form_key):
        task_name = st.text_input("Task Name")
        category = st.selectbox("Category", TASK_CATEGORIES)
        hours_per_week = st.number_input("Hours per week", min_value=0.0, max_value=168.0, step=0.5)
        stress = st.slider("Stress (1-5)", 1, 5, 3)
        urgency = st.slider("Urgency (1-5)", 1, 5, 3)
        importance = st.slider("Importance (1-5)", 1, 5, 3)
        mental_effort = st.slider("Mental Effort (1-5)", 1, 5, 3)
        add_task = st.form_submit_button("Add Task")

    if add_task:
        task_key = f"{task_name.strip()}_{category}_{hours_per_week}"
        if not task_name.strip() or task_name.strip().isdigit():
            st.warning("Please enter a valid task name.")
        elif hours_per_week == 0.0:
            st.warning("Please enter hours per week greater than 0.")
        elif task_key == st.session_state["last_added"]:
            st.warning("This task was already added.")
        else:
            st.session_state["tasks"].append({
                "name": task_name.strip(),
                "category": category,
                "hours_per_week": hours_per_week,
                "stress": stress,
                "urgency": urgency,
                "importance": importance,
                "mental_effort": mental_effort
            })
            st.session_state["last_added"] = task_key
            st.session_state["form_counter"] += 1
            st.rerun()

    if st.session_state["tasks"]:
        st.subheader("Your Tasks:")
        for i, task in enumerate(st.session_state["tasks"]):
            st.write(f"{i + 1}. {task['name']} — {task['category']}")

#Step 3: Results
if "tasks" in st.session_state and len(st.session_state["tasks"]) > 0:
    st.header("Step 3: Get Your Optimized Schedule")

    if st.button("Generate Schedule"):
        with st.spinner("Training model and predicting..."):
            combined = combine_features(
                st.session_state["averages"],
                st.session_state["tasks"]
            )

            if os.path.exists("models/model.pkl"):
                model, scaler = load_model()
            else:
                df = load_and_prepare("data/daily_activity_survey_data.xlsx")
                model, scaler = train_model(df)
                save_model(model, scaler)

            results = predict_tasks(model, scaler, combined)
            st.session_state["results"] = results

if "results" in st.session_state:
    st.subheader("Your Optimized To-Do List")
    st.markdown(f"Personality Type: **{st.session_state['category']}**")
    st.markdown("---")

    for i, task in enumerate(st.session_state["results"]):
        st.markdown(f"**{i + 1}. {task['name']}**")