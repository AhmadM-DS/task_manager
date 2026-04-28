"""
app.py

Streamlit UI for the Automated Task Management System.
Provides a web interface for the questionnaire, task input, and results.
"""

import os
import datetime
import streamlit as st
from src.questionnaire import questions, compute_profile
from src.combine_features import combine_features
from src.prepare_data import load_and_prepare
from src.model import train_model, predict_tasks, save_model, load_model
from src.constants import TASK_CATEGORIES

st.set_page_config(page_title="Task Manager", layout="wide")

st.title("Automated Task Manager")
st.markdown("Answer the questionnaire below to get a personalized task schedule.")

# ─────────────────────────────────────────────
# Step 1: Questionnaire
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Step 2: Category Ratings
# ─────────────────────────────────────────────
if "averages" in st.session_state:
    st.header("Step 2: Rate Your Task Categories")
    st.markdown(
        "Set default **Stress**, **Urgency**, **Importance**, and **Mental Effort** "
        "ratings for each category. These will auto-fill when you add a task — "
        "you can still adjust them per task if needed."
    )

    if "category_ratings" in st.session_state:
        if st.button("Re-rate Categories"):
            del st.session_state["category_ratings"]
            st.rerun()

    if "category_ratings" not in st.session_state:
        with st.form("category_ratings_form"):
            cat_ratings = {}
            for cat in TASK_CATEGORIES:
                st.subheader(cat)
                cols = st.columns(4)
                with cols[0]:
                    stress = st.slider("Stress", 1, 5, 3, key=f"cr_stress_{cat}")
                with cols[1]:
                    urgency = st.slider("Urgency", 1, 5, 3, key=f"cr_urgency_{cat}")
                with cols[2]:
                    importance = st.slider("Importance", 1, 5, 3, key=f"cr_importance_{cat}")
                with cols[3]:
                    mental_effort = st.slider("Mental Effort", 1, 5, 3, key=f"cr_mental_{cat}")
                cat_ratings[cat] = {
                    "stress": stress,
                    "urgency": urgency,
                    "importance": importance,
                    "mental_effort": mental_effort
                }
            save_ratings = st.form_submit_button("Save Category Ratings")

        if save_ratings:
            st.session_state["category_ratings"] = cat_ratings
            st.rerun()
    else:
        st.success("Category ratings saved. You can re-rate them anytime using the button above.")

# ─────────────────────────────────────────────
# Step 3: Task Input
# ─────────────────────────────────────────────
if "category_ratings" in st.session_state:
    st.header("Step 3: Enter Your To-Do List")

    if "tasks" not in st.session_state:
        st.session_state["tasks"] = []
    if "form_counter" not in st.session_state:
        st.session_state["form_counter"] = 0
    if "task_history" not in st.session_state:
        st.session_state["task_history"] = {}  # name -> last used task entry

    fc = st.session_state["form_counter"]

    # ── Previous task selector (outside form so it can trigger autofill) ──
    history_names = list(st.session_state["task_history"].keys())
    prev_options = ["— Type a new task —"] + history_names
    prev_selection = st.selectbox(
        "Use a previous task (or type a new one below)",
        prev_options,
        key=f"prev_select_{fc}"
    )

    # Determine autofill source
    if prev_selection != "— Type a new task —":
        autofill = st.session_state["task_history"][prev_selection]
    else:
        autofill = None

    # Resolve defaults for sliders: autofill > category defaults
    # Category selectbox is rendered inside the form below, so we need
    # a temporary category to look up category defaults before the form.
    # We use the autofill category if available, otherwise first TASK_CATEGORY.
    pre_category = autofill["category"] if autofill else TASK_CATEGORIES[0]
    cat_defaults = st.session_state["category_ratings"][pre_category]

    # Task type toggle outside form (controls which fields appear inside)
    default_type_index = 0
    if autofill:
        default_type_index = 1 if autofill["task_type"] == "block" else 0
    task_type = st.radio(
        "Task type",
        ["Weekly task (hours spread across the week)", "Block task (single uninterrupted session)"],
        index=default_type_index,
        key=f"task_type_{fc}"
    )
    is_block = task_type.startswith("Block")

    form_key = f"task_form_{fc}"

    with st.form(form_key):
        # Task Name
        default_name = autofill["name"] if autofill else ""
        task_name = st.text_input("Task Name", value=default_name)

        # Category (now under Task Name)
        default_cat_index = TASK_CATEGORIES.index(autofill["category"]) if autofill else 0
        selected_category = st.selectbox(
            "Category",
            TASK_CATEGORIES,
            index=default_cat_index,
            key=f"cat_select_{fc}"
        )

        if is_block:
            default_duration = autofill["duration_hours"] if autofill and autofill["task_type"] == "block" else 1.0
            duration_hours = st.number_input(
                "Duration (hours)", min_value=0.5, max_value=24.0,
                value=float(default_duration), step=0.5
            )
            st.markdown("**Specific date and time** *(optional — leave blank to auto-schedule)*")
            col_date, col_time = st.columns(2)
            with col_date:
                pinned_date = st.date_input("Date", value=None)
            with col_time:
                pinned_time = st.time_input("Start time", value=datetime.time(9, 0))
            hours_per_week = duration_hours
            max_session = None
        else:
            default_hpw = autofill["hours_per_week"] if autofill and autofill["task_type"] == "weekly" else 1.0
            default_max = (autofill.get("max_session") or 4.0) if autofill else 4.0
            hours_per_week = st.number_input(
                "Total hours per week", min_value=0.5, max_value=168.0,
                value=float(default_hpw), step=0.5
            )
            max_session = st.number_input(
                "Max session length (hours)", min_value=0.5, max_value=8.0,
                value=float(default_max), step=0.5,
                help="Each session will be capped at this length."
            )
            duration_hours = hours_per_week
            pinned_date = None
            pinned_time = None

        # Ratings — autofill from previous task if selected, else category defaults
        def_stress   = autofill["stress"]        if autofill else cat_defaults["stress"]
        def_urgency  = autofill["urgency"]       if autofill else cat_defaults["urgency"]
        def_import   = autofill["importance"]    if autofill else cat_defaults["importance"]
        def_mental   = autofill["mental_effort"] if autofill else cat_defaults["mental_effort"]

        st.markdown("**Ratings** *(pre-filled from previous task or category defaults — adjust if needed)*")
        rcols = st.columns(4)
        with rcols[0]:
            stress = st.slider("Stress (1-5)", 1, 5, def_stress)
        with rcols[1]:
            urgency = st.slider("Urgency (1-5)", 1, 5, def_urgency)
        with rcols[2]:
            importance = st.slider("Importance (1-5)", 1, 5, def_import)
        with rcols[3]:
            mental_effort = st.slider("Mental Effort (1-5)", 1, 5, def_mental)

        add_task = st.form_submit_button("Add Task")

    if add_task:
        if not task_name.strip() or task_name.strip().isdigit():
            st.warning("Please enter a valid task name.")
        elif duration_hours == 0.0:
            st.warning("Please enter a duration greater than 0.")
        else:
            task_entry = {
                "name": task_name.strip(),
                "category": selected_category,
                "task_type": "block" if is_block else "weekly",
                "hours_per_week": hours_per_week,
                "duration_hours": duration_hours,
                "stress": stress,
                "urgency": urgency,
                "importance": importance,
                "mental_effort": mental_effort,
                "max_session": max_session,
                "pinned_date": str(pinned_date) if (is_block and pinned_date) else None,
                "pinned_time": str(pinned_time) if (is_block and pinned_date) else None,
            }
            st.session_state["tasks"].append(task_entry)
            # Save to history (overwrites previous entry for same name)
            st.session_state["task_history"][task_entry["name"]] = task_entry
            st.session_state["form_counter"] += 1
            st.rerun()

    if st.session_state["tasks"]:
        st.subheader("Your Tasks:")
        for i, task in enumerate(st.session_state["tasks"]):
            tag = "📅 Block" if task["task_type"] == "block" else "🔁 Weekly"
            st.write(f"{i + 1}. {task['name']} — {task['category']} ({tag})")

# ─────────────────────────────────────────────
# Step 4: Generate Prioritized Schedule
# ─────────────────────────────────────────────
if "tasks" in st.session_state and len(st.session_state["tasks"]) > 0:
    st.header("Step 4: Generate Your Optimized Schedule")

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

# ─────────────────────────────────────────────
# Step 5: Calendar
# ─────────────────────────────────────────────
if "results" in st.session_state:
    st.header("Step 5: Calendar View")

    with st.expander("Set your available hours and date range", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            day_start = st.time_input("Day starts at", value=datetime.time(8, 0), key="day_start")
        with col2:
            day_end = st.time_input("Day ends at", value=datetime.time(22, 0), key="day_end")

        st.markdown("**Schedule date range** *(for auto-scheduled tasks)*")
        col3, col4 = st.columns(2)
        today = datetime.date.today()
        with col3:
            range_start = st.date_input("From", value=today, key="range_start")
        with col4:
            range_end = st.date_input(
                "To", value=today + datetime.timedelta(days=6), key="range_end"
            )
        build_cal = st.button("Build Calendar")

    if build_cal or "calendar_slots" in st.session_state:

        if build_cal:
            start_h = day_start.hour + day_start.minute / 60
            end_h = day_end.hour + day_end.minute / 60

            # Build free-slot tracker per date
            slots = {}
            d = range_start
            while d <= range_end:
                slots[d] = []
                d += datetime.timedelta(days=1)

            priority_order = {t["name"]: i for i, t in enumerate(st.session_state["results"])}

            pinned, unpinned_blocks, weekly_tasks = [], [], []
            for task in st.session_state["tasks"]:
                if task["task_type"] == "block":
                    if task["pinned_date"]:
                        pinned.append(task)
                    else:
                        unpinned_blocks.append(task)
                else:
                    weekly_tasks.append(task)

            unpinned_blocks.sort(key=lambda t: priority_order.get(t["name"], 999))
            weekly_tasks.sort(key=lambda t: priority_order.get(t["name"], 999))

            color_palette = [
                "#4F86C6", "#E07B54", "#5BAD72", "#A569BD",
                "#E4A835", "#48BFBF", "#D45F7A", "#7D8FA3"
            ]
            task_colors = {
                task["name"]: color_palette[i % len(color_palette)]
                for i, task in enumerate(st.session_state["tasks"])
            }

            def find_free_slot(slots, date, duration, s_h, e_h):
                occupied = sorted(slots[date], key=lambda x: x[0])
                cursor = s_h
                for (occ_s, occ_e, _) in occupied:
                    if cursor + duration <= occ_s:
                        return cursor
                    cursor = max(cursor, occ_e)
                if cursor + duration <= e_h:
                    return cursor
                return None

            def book(slots, date, s, e, name):
                slots[date].append((s, e, name))

            calendar_events = []

            # Place pinned blocks
            for task in pinned:
                d = datetime.date.fromisoformat(task["pinned_date"])
                tp = task["pinned_time"].split(":")
                s = int(tp[0]) + int(tp[1]) / 60
                e = s + task["duration_hours"]
                if d in slots:
                    book(slots, d, s, e, task["name"])
                calendar_events.append({
                    "name": task["name"], "date": d,
                    "start_h": s, "end_h": e,
                    "task_type": "block", "color": task_colors[task["name"]]
                })

            dates_in_range = sorted(slots.keys())

            # Place unpinned blocks
            for task in unpinned_blocks:
                placed = False
                for d in dates_in_range:
                    s = find_free_slot(slots, d, task["duration_hours"], start_h, end_h)
                    if s is not None:
                        e = s + task["duration_hours"]
                        book(slots, d, s, e, task["name"])
                        calendar_events.append({
                            "name": task["name"], "date": d,
                            "start_h": s, "end_h": e,
                            "task_type": "block", "color": task_colors[task["name"]]
                        })
                        placed = True
                        break
                if not placed:
                    st.warning(f"Could not fit '{task['name']}' in the selected date range.")

            # Place weekly tasks split into sessions
            for task in weekly_tasks:
                remaining = task["hours_per_week"]
                max_s = task.get("max_session") or 4.0
                for d in dates_in_range:
                    if remaining <= 0:
                        break
                    session = min(remaining, max_s)
                    s = find_free_slot(slots, d, session, start_h, end_h)
                    if s is not None:
                        e = s + session
                        book(slots, d, s, e, task["name"])
                        calendar_events.append({
                            "name": task["name"], "date": d,
                            "start_h": s, "end_h": e,
                            "task_type": "weekly", "color": task_colors[task["name"]]
                        })
                        remaining -= session
                if remaining > 0:
                    st.warning(
                        f"'{task['name']}' has {remaining:.1f} unscheduled hrs — "
                        f"try expanding your date range."
                    )

            st.session_state["calendar_slots"] = calendar_events
            st.session_state["task_colors"] = task_colors
            st.session_state["cal_range_start"] = range_start
            st.session_state["cal_range_end"] = range_end
            st.session_state["cal_day_start"] = start_h
            st.session_state["cal_day_end"] = end_h

        # ── Render calendar ──
        events = st.session_state["calendar_slots"]
        task_colors = st.session_state["task_colors"]
        cal_start = st.session_state["cal_range_start"]
        cal_end = st.session_state["cal_range_end"]
        cal_day_start = st.session_state["cal_day_start"]
        cal_day_end = st.session_state["cal_day_end"]

        view = st.radio("View", ["Weekly", "Monthly"], horizontal=True, key="cal_view")

        def fmt_hour(h):
            hour = int(h)
            minute = int(round((h - hour) * 60))
            suffix = "am" if hour < 12 else "pm"
            display = hour if hour <= 12 else hour - 12
            if display == 0:
                display = 12
            return f"{display}:{minute:02d}{suffix}"

        all_dates = []
        d = cal_start
        while d <= cal_end:
            all_dates.append(d)
            d += datetime.timedelta(days=1)

        events_by_date = {}
        for ev in events:
            events_by_date.setdefault(ev["date"], []).append(ev)

        if view == "Weekly":
            weeks = []
            week = []
            for d in all_dates:
                week.append(d)
                if len(week) == 7 or d == cal_end:
                    weeks.append(week)
                    week = []

            for week_dates in weeks:
                st.markdown(
                    f"**{week_dates[0].strftime('%b %d')} – "
                    f"{week_dates[-1].strftime('%b %d, %Y')}**"
                )
                day_cols = st.columns(len(week_dates))
                for col, day in zip(day_cols, week_dates):
                    with col:
                        is_today = day == datetime.date.today()
                        header_style = (
                            "background:#4F86C6;color:#fff;border-radius:6px 6px 0 0;"
                            if is_today else ""
                        )
                        st.markdown(
                            f"<div style='text-align:center;font-weight:600;font-size:12px;"
                            f"padding:5px;border-bottom:1px solid #ddd;{header_style}'>"
                            f"{day.strftime('%a')}<br>{day.strftime('%b %d')}</div>",
                            unsafe_allow_html=True
                        )
                        day_events = sorted(
                            events_by_date.get(day, []), key=lambda x: x["start_h"]
                        )
                        if not day_events:
                            st.markdown(
                                "<div style='color:#bbb;font-size:11px;text-align:center;"
                                "padding:10px'>—</div>",
                                unsafe_allow_html=True
                            )
                        for ev in day_events:
                            dur = ev["end_h"] - ev["start_h"]
                            icon = "📅" if ev["task_type"] == "block" else "🔁"
                            st.markdown(
                                f"<div style='background:{ev['color']};border-radius:5px;"
                                f"padding:5px 7px;margin:3px 0;font-size:11px;color:#fff'>"
                                f"<b>{icon} {ev['name']}</b><br>"
                                f"{fmt_hour(ev['start_h'])} – {fmt_hour(ev['end_h'])} "
                                f"({dur:.1f}h)</div>",
                                unsafe_allow_html=True
                            )
                st.markdown("---")

        else:  # Monthly
            import calendar as cal_mod
            months = {}
            for d in all_dates:
                key = (d.year, d.month)
                months.setdefault(key, []).append(d)

            for (year, month), month_dates in months.items():
                st.markdown(f"### {datetime.date(year, month, 1).strftime('%B %Y')}")
                h_cols = st.columns(7)
                for i, dn in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
                    h_cols[i].markdown(
                        f"<div style='text-align:center;font-weight:600;font-size:12px;"
                        f"padding:4px'>{dn}</div>",
                        unsafe_allow_html=True
                    )

                pad = month_dates[0].weekday()
                grid = [None] * pad + month_dates
                for row_start in range(0, len(grid), 7):
                    row = (grid[row_start:row_start+7] + [None]*7)[:7]
                    row_cols = st.columns(7)
                    for col, day in zip(row_cols, row):
                        with col:
                            if day is None:
                                st.markdown(
                                    "<div style='min-height:70px'></div>",
                                    unsafe_allow_html=True
                                )
                            else:
                                is_today = day == datetime.date.today()
                                border = "2px solid #4F86C6" if is_today else "1px solid #e0e0e0"
                                day_events = sorted(
                                    events_by_date.get(day, []), key=lambda x: x["start_h"]
                                )
                                content = (
                                    f"<div style='border:{border};border-radius:6px;"
                                    f"padding:4px;min-height:70px'>"
                                    f"<div style='font-size:12px;font-weight:600;"
                                    f"margin-bottom:2px'>{day.day}</div>"
                                )
                                for ev in day_events:
                                    dur = ev["end_h"] - ev["start_h"]
                                    content += (
                                        f"<div style='background:{ev['color']};"
                                        f"border-radius:3px;padding:2px 4px;margin:1px 0;"
                                        f"font-size:10px;color:#fff;white-space:nowrap;"
                                        f"overflow:hidden;text-overflow:ellipsis'>"
                                        f"{ev['name']} ({dur:.1f}h)</div>"
                                    )
                                content += "</div>"
                                st.markdown(content, unsafe_allow_html=True)
                st.markdown("---")

        # Legend
        st.markdown("**Legend**")
        leg_cols = st.columns(min(len(st.session_state["tasks"]), 4))
        for idx, task in enumerate(st.session_state["tasks"]):
            color = task_colors.get(task["name"], "#888")
            icon = "📅" if task["task_type"] == "block" else "🔁"
            with leg_cols[idx % 4]:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:6px;margin:3px 0'>"
                    f"<div style='width:12px;height:12px;border-radius:3px;"
                    f"background:{color};flex-shrink:0'></div>"
                    f"<span style='font-size:12px'>{icon} {task['name']}</span></div>",
                    unsafe_allow_html=True
                )
