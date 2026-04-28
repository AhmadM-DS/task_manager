"""
app.py

Streamlit UI for the Automated Task Management System.
Provides a web interface for the questionnaire, task input, and results.
"""

import os
import datetime
import streamlit as st
import streamlit.components.v1 as components
import json
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

    fc = st.session_state["form_counter"]
    cat_defaults = st.session_state["category_ratings"][TASK_CATEGORIES[0]]

    task_type = st.radio(
        "Task type",
        ["Weekly task (hours spread across the week)", "Block task (single uninterrupted session)"],
        key=f"task_type_{fc}"
    )
    is_block = task_type.startswith("Block")

    form_key = f"task_form_{fc}"

    with st.form(form_key):
        # Task Name
        task_name = st.text_input("Task Name")

        # Category (under Task Name)
        selected_category = st.selectbox(
            "Category",
            TASK_CATEGORIES,
            index=0,
            key=f"cat_select_{fc}"
        )

        if is_block:
            duration_hours = st.number_input(
                "Duration (hours)", min_value=0.5, max_value=24.0, value=1.0, step=0.5
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
            hours_per_week = st.number_input(
                "Total hours per week", min_value=0.5, max_value=168.0, value=1.0, step=0.5
            )
            max_session = st.number_input(
                "Max session length (hours)", min_value=0.5, max_value=8.0, value=4.0, step=0.5,
                help="Each session will be capped at this length."
            )
            duration_hours = hours_per_week
            pinned_date = None
            pinned_time = None

        st.markdown("**Ratings** *(pre-filled from category defaults — adjust if needed)*")
        rcols = st.columns(4)
        with rcols[0]:
            stress = st.slider("Stress (1-5)", 1, 5, cat_defaults["stress"])
        with rcols[1]:
            urgency = st.slider("Urgency (1-5)", 1, 5, cat_defaults["urgency"])
        with rcols[2]:
            importance = st.slider("Importance (1-5)", 1, 5, cat_defaults["importance"])
        with rcols[3]:
            mental_effort = st.slider("Mental Effort (1-5)", 1, 5, cat_defaults["mental_effort"])

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
            st.session_state["form_counter"] += 1
            st.rerun()

    if st.session_state["tasks"]:
        st.subheader("Your Tasks:")

        if "editing_task_index" not in st.session_state:
            st.session_state["editing_task_index"] = None

        for i, task in enumerate(st.session_state["tasks"]):
            tag = "📅 Block" if task["task_type"] == "block" else "🔁 Weekly"
            col_label, col_edit, col_del = st.columns([6, 1, 1])
            with col_label:
                st.write(f"{i + 1}. {task['name']} — {task['category']} ({tag})")
            with col_edit:
                if st.button("✏️", key=f"edit_task_{i}", help="Edit scheduling fields"):
                    st.session_state["editing_task_index"] = i
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_task_{i}", help="Delete this task"):
                    st.session_state["tasks"].pop(i)
                    if st.session_state["editing_task_index"] == i:
                        st.session_state["editing_task_index"] = None
                    st.rerun()

            # Inline edit form for this task
            if st.session_state["editing_task_index"] == i:
                with st.form(f"edit_form_{i}"):
                    st.markdown(f"**Editing: {task['name']}**")
                    if task["task_type"] == "block":
                        new_duration = st.number_input(
                            "Duration (hours)", min_value=0.5, max_value=24.0,
                            value=float(task.get("duration_hours", 1.0)), step=0.5
                        )
                        st.markdown("**Date and time** *(leave date blank to auto-schedule)*")
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            existing_date = datetime.date.fromisoformat(task["pinned_date"]) if task.get("pinned_date") else None
                            new_date = st.date_input("Date", value=existing_date)
                        with ec2:
                            existing_time = datetime.time.fromisoformat(task["pinned_time"]) if task.get("pinned_time") else datetime.time(9, 0)
                            new_time = st.time_input("Start time", value=existing_time)
                    else:
                        new_duration = st.number_input(
                            "Total hours per week", min_value=0.5, max_value=168.0,
                            value=float(task.get("hours_per_week", 1.0)), step=0.5
                        )
                        new_max = st.number_input(
                            "Max session length (hours)", min_value=0.5, max_value=8.0,
                            value=float(task.get("max_session") or 4.0), step=0.5
                        )

                    save_col, cancel_col = st.columns(2)
                    with save_col:
                        save_edit = st.form_submit_button("💾 Save")
                    with cancel_col:
                        cancel_edit = st.form_submit_button("✖ Cancel")

                if save_edit:
                    if task["task_type"] == "block":
                        st.session_state["tasks"][i]["duration_hours"] = new_duration
                        st.session_state["tasks"][i]["hours_per_week"] = new_duration
                        st.session_state["tasks"][i]["pinned_date"] = str(new_date) if new_date else None
                        st.session_state["tasks"][i]["pinned_time"] = str(new_time) if new_date else None
                    else:
                        st.session_state["tasks"][i]["hours_per_week"] = new_duration
                        st.session_state["tasks"][i]["duration_hours"] = new_duration
                        st.session_state["tasks"][i]["max_session"] = new_max
                    if "calendar_slots" in st.session_state:
                        del st.session_state["calendar_slots"]
                    st.session_state["editing_task_index"] = None
                    st.rerun()

                if cancel_edit:
                    st.session_state["editing_task_index"] = None
                    st.rerun()

        if st.button("🗑️ Delete All Tasks"):
            st.session_state["tasks"] = []
            st.session_state["editing_task_index"] = None
            if "results" in st.session_state:
                del st.session_state["results"]
            if "calendar_slots" in st.session_state:
                del st.session_state["calendar_slots"]
            st.rerun()

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
        if (day_end.hour + day_end.minute / 60) <= (day_start.hour + day_start.minute / 60):
            st.info("🌙 Overnight schedule detected — your day will be scheduled across midnight.")

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

            # Handle overnight schedules (e.g. 11am start, 4am end next day)
            # If end is before or equal to start, the window crosses midnight.
            # Represent as continuous hours: e.g. 4am becomes 28 (24+4).
            if end_h <= start_h:
                end_h += 24

            # Build free-slot tracker per date.
            # Add one extra day as overflow buffer for overnight schedules.
            slots = {}
            d = range_start
            end_buffer = range_end + datetime.timedelta(days=1)
            while d <= end_buffer:
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

            TRANSITION = 0.5        # 30-minute gap between tasks
            ERRANDS_CUTOFF = 22.0   # Errands must end by 10pm

            def get_effective_end(task_category, e_h):
                """Return the applicable end hour for a task, respecting curfews."""
                if task_category == "Errands":
                    return min(e_h, ERRANDS_CUTOFF)
                return e_h

            def find_free_slot(slots, date, duration, s_h, e_h):
                """Find first free slot on date with transition buffer."""
                # Include transition buffer in occupied ranges
                occupied = sorted(
                    [(occ_s, occ_e + TRANSITION, n) for (occ_s, occ_e, n) in slots[date]],
                    key=lambda x: x[0]
                )
                cursor = s_h
                for (occ_s, occ_e_buf, _) in occupied:
                    if cursor + duration <= occ_s:
                        return cursor
                    cursor = max(cursor, occ_e_buf)
                if cursor + duration <= e_h:
                    return cursor
                return None

            def day_load(slots, date):
                """Total booked hours on a date (excluding transition buffers)."""
                return sum(e - s for (s, e, _) in slots[date])

            def book(slots, date, s, e, name):
                slots[date].append((s, e, name))

            calendar_events = []

            # Place pinned blocks (user-specified date/time — no redistribution)
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

            dates_in_range = sorted(
                [d for d in slots.keys() if d <= range_end]
            )

            # Place unpinned block tasks — pick the least-loaded available day
            for task in unpinned_blocks:
                eff_end = get_effective_end(task["category"], end_h)
                # Sort dates by current load so task lands on lightest day
                candidates = sorted(dates_in_range, key=lambda d: day_load(slots, d))
                placed = False
                for d in candidates:
                    s = find_free_slot(slots, d, task["duration_hours"], start_h, eff_end)
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

            # Place weekly tasks — distribute sessions evenly across the week
            for task in weekly_tasks:
                remaining = task["hours_per_week"]
                max_s = task.get("max_session") or 4.0
                eff_end = get_effective_end(task["category"], end_h)
                while remaining > 0:
                    session = min(remaining, max_s)
                    # Pick the least-loaded day that has room
                    candidates = sorted(dates_in_range, key=lambda d: day_load(slots, d))
                    placed_session = False
                    for d in candidates:
                        s = find_free_slot(slots, d, session, start_h, eff_end)
                        if s is not None:
                            e = s + session
                            book(slots, d, s, e, task["name"])
                            calendar_events.append({
                                "name": task["name"], "date": d,
                                "start_h": s, "end_h": e,
                                "task_type": "weekly", "color": task_colors[task["name"]]
                            })
                            remaining -= session
                            placed_session = True
                            break
                    if not placed_session:
                        st.warning(
                            f"'{task['name']}' has {remaining:.1f} unscheduled hrs — "
                            f"try expanding your date range."
                        )
                        break

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
            # Normalize hours > 24 back to clock time (e.g. 28 -> 4am)
            h = h % 24
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

            PX_PER_HOUR = 64
            SNAP = 0.5  # 30-minute snap
            today = datetime.date.today()

            # Handle drag-drop result coming back from JS
            if "pending_drag" in st.session_state:
                drag = st.session_state.pop("pending_drag")
                # Find event in calendar_slots and update it
                updated = False
                for ev in st.session_state["calendar_slots"]:
                    if ev["name"] == drag["name"] and abs(ev["start_h"] - drag["old_start"]) < 0.01 and str(ev["date"]) == drag["old_date"]:
                        new_date = datetime.date.fromisoformat(drag["new_date"])
                        new_start = drag["new_start"]
                        dur = ev["end_h"] - ev["start_h"]
                        ev["date"] = new_date
                        ev["start_h"] = new_start
                        ev["end_h"] = new_start + dur
                        updated = True
                        break
                if updated:
                    st.rerun()

            for week_idx, week_dates in enumerate(weeks):
                # Serialize events for this week to JSON for JS
                week_events = []
                for day in week_dates:
                    for ev in events_by_date.get(day, []):
                        week_events.append({
                            "name": ev["name"],
                            "date": str(ev["date"]),
                            "start_h": ev["start_h"],
                            "end_h": ev["end_h"],
                            "color": ev["color"],
                            "task_type": ev["task_type"],
                        })

                week_dates_str = [str(d) for d in week_dates]
                week_labels = [f"{d.strftime('%a')}<br>{d.strftime('%b %d')}" for d in week_dates]
                is_today_flags = [d == today for d in week_dates]
                total_hours = cal_day_end - cal_day_start
                grid_height = int(total_hours * PX_PER_HOUR)

                # Hour tick labels every 2 hours
                hour_ticks = []
                h = cal_day_start
                while h <= cal_day_end:
                    hour_ticks.append({"h": h, "label": fmt_hour(h % 24)})
                    h += 2

                component_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: sans-serif; }}
  body {{ background: transparent; }}
  #cal-wrap {{ user-select: none; }}
  .header-row {{ display: flex; margin-left: 52px; margin-bottom: 4px; }}
  .day-header {{ flex: 1; text-align: center; padding: 5px 2px; font-weight: 600;
                 font-size: 12px; border-radius: 4px; margin: 0 1px; line-height:1.4; }}
  .grid-row {{ display: flex; }}
  .time-col {{ width: 52px; flex-shrink: 0; position: relative; height: {grid_height}px; }}
  .time-label {{ position: absolute; right: 6px; font-size: 10px; color: #888;
                 transform: translateY(-50%); white-space: nowrap; }}
  .days-area {{ flex: 1; display: flex; position: relative; height: {grid_height}px; }}
  .day-col {{ flex: 1; position: relative; border-left: 1px solid #e0e0e0;
              height: {grid_height}px; margin: 0 1px; }}
  .hour-line {{ position: absolute; left: 0; right: 0; border-top: 1px solid #eee; pointer-events: none; }}
  .event-block {{ position: absolute; left: 2px; right: 2px; border-radius: 5px;
                  padding: 3px 5px; color: #fff; overflow: hidden; cursor: grab;
                  line-height: 1.3; transition: box-shadow 0.1s; font-size: 10px; }}
  .event-block:active {{ cursor: grabbing; }}
  .event-block.dragging {{ opacity: 0.5; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
  .drop-highlight {{ background: rgba(79,134,198,0.15) !important; }}
  #conflict-msg {{ color: #c0392b; font-size: 12px; margin-top: 6px;
                   min-height: 18px; text-align: center; }}
</style>
</head>
<body>
<div id="cal-wrap">
  <div class="header-row" id="header-row"></div>
  <div class="grid-row">
    <div class="time-col" id="time-col"></div>
    <div class="days-area" id="days-area"></div>
  </div>
  <div id="conflict-msg"></div>
</div>
<script>
const PX_PER_HOUR = {PX_PER_HOUR};
const SNAP = {SNAP};
const DAY_START = {cal_day_start};
const DAY_END   = {cal_day_end};
const GRID_H    = {grid_height};

const WEEK_DATES  = {json.dumps(week_dates_str)};
const WEEK_LABELS = {json.dumps(week_labels)};
const IS_TODAY    = {json.dumps(is_today_flags)};
const HOUR_TICKS  = {json.dumps(hour_ticks)};
let events = {json.dumps(week_events)};

// ── Build headers ──
const headerRow = document.getElementById('header-row');
WEEK_DATES.forEach((_, i) => {{
  const d = document.createElement('div');
  d.className = 'day-header';
  d.style.background = IS_TODAY[i] ? '#4F86C6' : '#f5f5f5';
  d.style.color       = IS_TODAY[i] ? '#fff'    : '#333';
  d.innerHTML = WEEK_LABELS[i];
  headerRow.appendChild(d);
}});

// ── Build time column ──
const timeCol = document.getElementById('time-col');
timeCol.style.height = GRID_H + 'px';
HOUR_TICKS.forEach(tick => {{
  const top = (tick.h - DAY_START) * PX_PER_HOUR;
  const lbl = document.createElement('div');
  lbl.className = 'time-label';
  lbl.style.top = top + 'px';
  lbl.textContent = tick.label;
  timeCol.appendChild(lbl);
}});

// ── Build day columns ──
const daysArea = document.getElementById('days-area');
const dayCols = [];
WEEK_DATES.forEach((dateStr, di) => {{
  const col = document.createElement('div');
  col.className = 'day-col';
  col.dataset.date = dateStr;
  col.dataset.di   = di;
  // Hour lines
  HOUR_TICKS.forEach(tick => {{
    const line = document.createElement('div');
    line.className = 'hour-line';
    line.style.top = ((tick.h - DAY_START) * PX_PER_HOUR) + 'px';
    col.appendChild(line);
  }});
  daysArea.appendChild(col);
  dayCols.push(col);
}});

// ── Helpers ──
function snapH(h) {{
  return Math.round(h / SNAP) * SNAP;
}}

function fmtHour(h) {{
  h = ((h % 24) + 24) % 24;
  const hr = Math.floor(h);
  const mn = Math.round((h - hr) * 60);
  const suffix = hr < 12 ? 'am' : 'pm';
  let disp = hr <= 12 ? hr : hr - 12;
  if (disp === 0) disp = 12;
  return disp + ':' + String(mn).padStart(2,'0') + suffix;
}}

function hasConflict(evName, dateStr, newStart, dur) {{
  const newEnd = newStart + dur;
  const TRANS = 0.5;
  for (const ev of events) {{
    if (ev.name === evName) continue;
    if (ev.date !== dateStr) continue;
    const os = ev.start_h;
    const oe = ev.end_h + TRANS;
    // overlap check with transition buffer
    if (newStart < oe && newEnd + TRANS > os) return true;
  }}
  return false;
}}

// ── Render all events ──
function renderEvents() {{
  // Remove existing event blocks
  dayCols.forEach(col => {{
    col.querySelectorAll('.event-block').forEach(el => el.remove());
  }});

  events.forEach((ev, evIdx) => {{
    const di = WEEK_DATES.indexOf(ev.date);
    if (di < 0) return;
    const col = dayCols[di];

    const visStart = Math.max(ev.start_h, DAY_START);
    const visEnd   = Math.min(ev.end_h,   DAY_END);
    if (visEnd <= visStart) return;

    const topPx    = (visStart - DAY_START) * PX_PER_HOUR;
    const heightPx = Math.max((visEnd - visStart) * PX_PER_HOUR - 2, 18);
    const dur      = ev.end_h - ev.start_h;
    const icon     = ev.task_type === 'block' ? '📅' : '🔁';

    const block = document.createElement('div');
    block.className   = 'event-block';
    block.style.top   = topPx + 'px';
    block.style.height= heightPx + 'px';
    block.style.background = ev.color;
    block.dataset.evIdx = evIdx;

    if (heightPx >= 28) {{
      block.innerHTML = `<b>${{icon}} ${{ev.name}}</b><br>
        <span style="font-size:9px">${{fmtHour(ev.start_h)}}–${{fmtHour(ev.end_h)}} (${{dur.toFixed(1)}}h)</span>`;
    }} else {{
      block.innerHTML = `<span>${{ev.name}}</span>`;
    }}

    // ── Drag logic ──
    block.addEventListener('mousedown', onMouseDown);
    col.appendChild(block);
  }});
}}

// ── Drag state ──
let drag = null;

function onMouseDown(e) {{
  if (e.button !== 0) return;
  e.preventDefault();
  const block   = e.currentTarget;
  const evIdx   = parseInt(block.dataset.evIdx);
  const ev      = events[evIdx];
  const colEl   = block.closest('.day-col');
  const colRect = colEl.getBoundingClientRect();
  const blockRect = block.getBoundingClientRect();
  const offsetY = e.clientY - blockRect.top;  // click offset within block

  block.classList.add('dragging');

  drag = {{ evIdx, ev, block, offsetY,
             origDate: ev.date, origStart: ev.start_h }};

  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mouseup',   onMouseUp);
}}

function onMouseMove(e) {{
  if (!drag) return;
  const daysRect = daysArea.getBoundingClientRect();

  // Which day column are we over?
  let targetDi = -1;
  let targetDate = null;
  dayCols.forEach((col, di) => {{
    const r = col.getBoundingClientRect();
    if (e.clientX >= r.left && e.clientX <= r.right) {{
      targetDi   = di;
      targetDate = WEEK_DATES[di];
    }}
    col.classList.remove('drop-highlight');
  }});

  if (targetDi >= 0) {{
    dayCols[targetDi].classList.add('drop-highlight');
    const colRect = dayCols[targetDi].getBoundingClientRect();
    const rawH = (e.clientY - drag.offsetY - colRect.top) / PX_PER_HOUR + DAY_START;
    const snappedStart = snapH(rawH);
    const dur = drag.ev.end_h - drag.ev.start_h;

    // Move block visually
    const topPx = (Math.max(snappedStart, DAY_START) - DAY_START) * PX_PER_HOUR;
    drag.block.style.top = topPx + 'px';

    drag._targetDi    = targetDi;
    drag._targetDate  = targetDate;
    drag._newStart    = snappedStart;
  }}
}}

function onMouseUp(e) {{
  if (!drag) return;
  document.removeEventListener('mousemove', onMouseMove);
  document.removeEventListener('mouseup',   onMouseUp);
  drag.block.classList.remove('dragging');
  dayCols.forEach(col => col.classList.remove('drop-highlight'));

  const conflictMsg = document.getElementById('conflict-msg');
  conflictMsg.textContent = '';

  if (drag._targetDate != null) {{
    const dur      = drag.ev.end_h - drag.ev.start_h;
    const newStart = drag._newStart;
    const newEnd   = newStart + dur;

    // Boundary check
    if (newStart < DAY_START || newEnd > DAY_END) {{
      conflictMsg.textContent = '⚠️ Task would go outside your available hours — move cancelled.';
      renderEvents();
      drag = null;
      return;
    }}

    // Conflict check
    if (hasConflict(drag.ev.name, drag._targetDate, newStart, dur)) {{
      conflictMsg.textContent = '⚠️ That slot overlaps another task — move cancelled.';
      renderEvents();
      drag = null;
      return;
    }}

    // Apply move
    const oldDate  = drag.ev.date;
    const oldStart = drag.ev.start_h;
    events[drag.evIdx].date    = drag._targetDate;
    events[drag.evIdx].start_h = newStart;
    events[drag.evIdx].end_h   = newStart + dur;

    renderEvents();

    // Send update back to Streamlit via query param trick
    const payload = JSON.stringify({{
      name:      drag.ev.name,
      old_date:  oldDate,
      old_start: oldStart,
      new_date:  drag._targetDate,
      new_start: newStart
    }});
    // Post to Streamlit parent
    window.parent.postMessage({{type: 'streamlit:setComponentValue', value: payload}}, '*');
  }} else {{
    renderEvents();
  }}
  drag = null;
}}

renderEvents();
</script>
</body>
</html>
"""
                result = components.html(component_html, height=grid_height + 80, scrolling=False)

                # Handle drag result returned from JS component
                if result and isinstance(result, str):
                    try:
                        drag_data = json.loads(result)
                        st.session_state["pending_drag"] = drag_data
                        st.rerun()
                    except Exception:
                        pass

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

        # ── Calendar Edit Panel ──
        st.markdown("---")
        st.subheader("✏️ Edit a Scheduled Task")
        task_names = [t["name"] for t in st.session_state["tasks"]]
        cal_edit_name = st.selectbox(
            "Select a task to edit its scheduling",
            ["— Select a task —"] + task_names,
            key="cal_edit_select"
        )

        if cal_edit_name != "— Select a task —":
            cal_edit_idx = next(
                (i for i, t in enumerate(st.session_state["tasks"]) if t["name"] == cal_edit_name), None
            )
            if cal_edit_idx is not None:
                cal_task = st.session_state["tasks"][cal_edit_idx]
                with st.form("cal_edit_form"):
                    st.markdown(f"**{cal_task['name']}** — {cal_task['category']}")
                    if cal_task["task_type"] == "block":
                        cal_new_duration = st.number_input(
                            "Duration (hours)", min_value=0.5, max_value=24.0,
                            value=float(cal_task.get("duration_hours", 1.0)), step=0.5
                        )
                        st.markdown("**Date and time** *(leave date blank to auto-schedule)*")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            cal_existing_date = datetime.date.fromisoformat(cal_task["pinned_date"]) if cal_task.get("pinned_date") else None
                            cal_new_date = st.date_input("Date", value=cal_existing_date, key="cal_edit_date")
                        with cc2:
                            cal_existing_time = datetime.time.fromisoformat(cal_task["pinned_time"]) if cal_task.get("pinned_time") else datetime.time(9, 0)
                            cal_new_time = st.time_input("Start time", value=cal_existing_time, key="cal_edit_time")
                    else:
                        cal_new_duration = st.number_input(
                            "Total hours per week", min_value=0.5, max_value=168.0,
                            value=float(cal_task.get("hours_per_week", 1.0)), step=0.5
                        )
                        cal_new_max = st.number_input(
                            "Max session length (hours)", min_value=0.5, max_value=8.0,
                            value=float(cal_task.get("max_session") or 4.0), step=0.5
                        )

                    cs1, cs2 = st.columns(2)
                    with cs1:
                        cal_save = st.form_submit_button("💾 Save & Rebuild Calendar")
                    with cs2:
                        cal_delete = st.form_submit_button("🗑️ Delete This Task")

                if cal_save:
                    if cal_task["task_type"] == "block":
                        st.session_state["tasks"][cal_edit_idx]["duration_hours"] = cal_new_duration
                        st.session_state["tasks"][cal_edit_idx]["hours_per_week"] = cal_new_duration
                        st.session_state["tasks"][cal_edit_idx]["pinned_date"] = str(cal_new_date) if cal_new_date else None
                        st.session_state["tasks"][cal_edit_idx]["pinned_time"] = str(cal_new_time) if cal_new_date else None
                    else:
                        st.session_state["tasks"][cal_edit_idx]["hours_per_week"] = cal_new_duration
                        st.session_state["tasks"][cal_edit_idx]["duration_hours"] = cal_new_duration
                        st.session_state["tasks"][cal_edit_idx]["max_session"] = cal_new_max
                    if "calendar_slots" in st.session_state:
                        del st.session_state["calendar_slots"]
                    st.rerun()

                if cal_delete:
                    st.session_state["tasks"].pop(cal_edit_idx)
                    if "calendar_slots" in st.session_state:
                        del st.session_state["calendar_slots"]
                    if "results" in st.session_state:
                        del st.session_state["results"]
                    st.rerun()
