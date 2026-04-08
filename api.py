from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODEL ---------------- #

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- LOAD CALENDAR ---------------- #

import os
calanderURL= 'https://raw.githubusercontent.com/bhavinSOL/TATA-Attendance/refs/heads/main/public/2026_calander.csv'
attandanceURL='https://raw.githubusercontent.com/bhavinSOL/TATA-Attendance/refs/heads/main/public/attendance.csv'

calendar_df = pd.read_csv(calanderURL)
calendar_df["date"] = pd.to_datetime(calendar_df["date"], dayfirst=True)

# ---------------- LOAD ATTENDANCE HISTORY ---------------- #

attendance_df = pd.read_csv(attandanceURL)
attendance_df["date"] = pd.to_datetime(attendance_df["date"], dayfirst=True)
attendance_df = attendance_df.sort_values("date").reset_index(drop=True)

# ---------------- NORMAL DAY STATISTICS (for realistic bounds) ---------------- #

normal_days = attendance_df[
    (attendance_df["is_holiday"] == 0) &
    (attendance_df["is_festival"] == 0) &
    (attendance_df["absent_percent"] < 50)  # exclude outlier rows
]
NORMAL_MEAN = float(normal_days["absent_percent"].mean())    # ~12.75%
NORMAL_STD  = float(normal_days["absent_percent"].std())     # ~3-4%
NORMAL_MAX_CAP = round(NORMAL_MEAN + 1.5 * NORMAL_STD, 2)   # realistic upper bound
NORMAL_MIN_CAP = max(round(NORMAL_MEAN - 2 * NORMAL_STD, 2), 2.0)  # realistic lower bound

print(f"📊 Normal day stats: mean={NORMAL_MEAN:.2f}%, std={NORMAL_STD:.2f}%, cap=[{NORMAL_MIN_CAP}, {NORMAL_MAX_CAP}]")

# ---------------- RECENT ATTENDANCE HELPERS ---------------- #

def safe_float(value):
    """Convert a value to float, returning None if it's NaN or missing."""
    if value is None:
        return None
    f = float(value)
    if np.isnan(f):
        return None
    return f


def get_previous_days_attendance(selected_date, num_days=3, only_working_days=False):
    """
    Get the actual attendance (absent_percent) for the previous N days.
    If only_working_days=True, skip holidays/festivals/Sundays to avoid
    inflating the average with 90%+ values.
    Returns a list of absent_percent values and their average.
    """
    selected_date = pd.to_datetime(selected_date)
    results = []
    checked = 0
    i = 1

    # Look back further if filtering working days only
    while len(results) < num_days and checked < num_days + 10:
        prev_date = selected_date - timedelta(days=i)
        i += 1
        checked += 1
        row = attendance_df[attendance_df["date"] == prev_date]
        if len(row) > 0:
            r = row.iloc[0]
            val = safe_float(r["absent_percent"])
            if val is None:
                continue
            # Skip holidays/Sundays if we only want working day trends
            if only_working_days and (r.get("is_holiday", 0) == 1 or r.get("is_festival", 0) == 1):
                continue
            results.append({
                "date": str(prev_date.date()),
                "absent_percent": val
            })

    avg = safe_float(np.mean([r["absent_percent"] for r in results])) if results else None
    return results, avg


def get_previous_week_attendance(selected_date, only_working_days=False):
    """
    Get the actual attendance for the same day last week and
    the average of the entire previous week (7 days before selected_date).
    If only_working_days=True, excludes holidays/festivals from average.
    """
    selected_date = pd.to_datetime(selected_date)

    # Same day last week
    same_day_last_week = selected_date - timedelta(days=7)
    same_day_row = attendance_df[attendance_df["date"] == same_day_last_week]
    same_day_absent = safe_float(same_day_row.iloc[0]["absent_percent"]) if len(same_day_row) > 0 else None

    # Full previous week (7 days ending the day before selected_date)
    week_start = selected_date - timedelta(days=7)
    week_end = selected_date - timedelta(days=1)
    week_data = attendance_df[
        (attendance_df["date"] >= week_start) & (attendance_df["date"] <= week_end)
    ]

    week_records = []
    working_day_values = []
    for _, row in week_data.iterrows():
        val = safe_float(row["absent_percent"])
        if val is not None:
            week_records.append({
                "date": str(row["date"].date()),
                "absent_percent": val
            })
            # Collect only working day values for the average
            if only_working_days:
                if row.get("is_holiday", 0) == 0 and row.get("is_festival", 0) == 0:
                    working_day_values.append(val)
            else:
                working_day_values.append(val)

    week_avg = safe_float(np.mean(working_day_values)) if working_day_values else None

    return {
        "same_day_last_week": {
            "date": str(same_day_last_week.date()),
            "absent_percent": same_day_absent
        },
        "previous_week_records": week_records,
        "previous_week_avg": week_avg
    }


def hybrid_prediction(model_pred, prev_3_day_avg, prev_week_avg, is_normal_day=True):
    """
    Combine model prediction with recent actual attendance trends.
    Weights:
      - 60% model prediction (calendar/holiday features)
      - 20% previous 3-day average (recent trend)
      - 20% previous week average (weekly pattern)
    Falls back to model-only if actual data is unavailable.
    For normal working days, clips the result to realistic bounds.
    """
    if prev_3_day_avg is not None and prev_week_avg is not None:
        result = 0.6 * model_pred + 0.2 * prev_3_day_avg + 0.2 * prev_week_avg
    elif prev_3_day_avg is not None:
        result = 0.7 * model_pred + 0.3 * prev_3_day_avg
    elif prev_week_avg is not None:
        result = 0.7 * model_pred + 0.3 * prev_week_avg
    else:
        result = model_pred

    # Clip normal working days to realistic range
    if is_normal_day:
        result = max(NORMAL_MIN_CAP, min(result, NORMAL_MAX_CAP))

    return result


# ---------------- FEATURE BUILDER ---------------- #

def build_features(selected_date):

    selected_date = pd.to_datetime(selected_date)

    cal_row = calendar_df[calendar_df["date"] == selected_date]

    is_holiday = 0
    is_festival = 0
    festival_weight = 0

    if len(cal_row) > 0:
        is_holiday = cal_row.iloc[0].get("is_holiday", 0)
        is_festival = cal_row.iloc[0].get("is_festival", 0)
        festival_weight = cal_row.iloc[0].get("festival_weight", 0)

    # Convert Python weekday (Mon=0..Sun=6) to CSV format (Sun=0..Sat=6)
    # Python: Mon=0,Tue=1,Wed=2,Thu=3,Fri=4,Sat=5,Sun=6
    # CSV:    Sun=0,Mon=1,Tue=2,Wed=3,Thu=4,Fri=5,Sat=6
    py_weekday = selected_date.weekday()  # Mon=0 .. Sun=6
    day_of_week = (py_weekday + 1) % 7     # Sun=0 .. Sat=6

    week_number = selected_date.isocalendar()[1]
    month = selected_date.month

    features = np.array([[
        day_of_week,
        week_number,
        month,
        is_holiday,
        is_festival,
        festival_weight
    ]])

    return features


# ---------------- DAY ---------------- #

@app.route("/predict/day")
def predict_day():
    date = request.args.get("date")
    features = build_features(date)
    model_prediction = model.predict(features)[0]

    # Check if this is a normal working day (not holiday/festival)
    selected_date = pd.to_datetime(date)
    cal_row = calendar_df[calendar_df["date"] == selected_date]
    is_holiday = int(cal_row.iloc[0].get("is_holiday", 0)) if len(cal_row) > 0 else 0
    is_festival = int(cal_row.iloc[0].get("is_festival", 0)) if len(cal_row) > 0 else 0
    is_normal_day = (is_holiday == 0 and is_festival == 0)

    # Holidays/Sundays/Off days: pure model prediction, no hybrid
    if not is_normal_day:
        return jsonify({
            "date": date,
            "predicted_absentees_percentage": f"{round(float(model_prediction), 2)}%",
            "model_only_prediction": f"{round(float(model_prediction), 2)}%",
            "is_normal_day": False,
            "normal_day_cap": None,
            "recent_context": None
        })

    # Normal working days: hybrid with prev actual data + cap
    prev_3_days, prev_3_day_avg = get_previous_days_attendance(
        date, num_days=3, only_working_days=True
    )
    prev_week_info = get_previous_week_attendance(
        date, only_working_days=True
    )
    prev_week_avg = prev_week_info["previous_week_avg"]

    final_prediction = hybrid_prediction(
        model_prediction, prev_3_day_avg, prev_week_avg, is_normal_day=True
    )

    return jsonify({
        "date": date,
        "predicted_absentees_percentage": f"{round(float(final_prediction), 2)}%",
        "model_only_prediction": f"{round(float(model_prediction), 2)}%",
        "is_normal_day": is_normal_day,
        "normal_day_cap": NORMAL_MAX_CAP if is_normal_day else None,
        "recent_context": {
            "previous_3_days": prev_3_days,
            "previous_3_day_avg": f"{round(prev_3_day_avg, 2)}%" if prev_3_day_avg is not None else None,
            "previous_week": prev_week_info
        }
    })


# ---------------- HELPER: cap prediction for a date ---------------- #

def smart_predict(date):
    """
    Full hybrid prediction for any single date:
    model + prev 3 working days + prev week actual data + normal-day cap.
    Used by all endpoints (day, week, month, range).
    """
    features = build_features(date)
    model_pred = model.predict(features)[0]

    dt = pd.to_datetime(date)
    cal_row = calendar_df[calendar_df["date"] == dt]
    is_holiday = int(cal_row.iloc[0].get("is_holiday", 0)) if len(cal_row) > 0 else 0
    is_festival = int(cal_row.iloc[0].get("is_festival", 0)) if len(cal_row) > 0 else 0
    is_normal_day = (is_holiday == 0 and is_festival == 0)

    # Holidays/Sundays/Off days: use pure model prediction (no hybrid, no cap)
    if not is_normal_day:
        return model_pred

    # Normal working days: hybrid with prev actual data + cap
    _, prev_3_day_avg = get_previous_days_attendance(
        date, num_days=3, only_working_days=True
    )
    prev_week_info = get_previous_week_attendance(
        date, only_working_days=True
    )
    prev_week_avg = prev_week_info["previous_week_avg"]

    result = hybrid_prediction(model_pred, prev_3_day_avg, prev_week_avg, is_normal_day=True)

    return result


# ---------------- WEEK ---------------- #

@app.route("/predict/week")
def predict_week():
    start_date = request.args.get("start_date")
    start = pd.to_datetime(start_date)

    total = 0

    for i in range(7):
        current = start + timedelta(days=i)
        total += smart_predict(current)

    avg = total / 7

    return jsonify({
        "week_start": start_date,
        "average_week_absentees_percentage": f"{round(float(avg), 2)}%"
    })


# ---------------- MONTH ---------------- #

@app.route("/predict/month")
def predict_month():
    year = int(request.args.get("year"))
    month = int(request.args.get("month"))

    start = datetime(year, month, 1)
    end = (start + pd.offsets.MonthEnd(0)).to_pydatetime()

    total = 0
    count = 0
    current = start

    while current <= end:
        total += smart_predict(current)
        count += 1
        current += timedelta(days=1)

    avg = total / count

    return jsonify({
        "year": year,
        "month": month,
        "average_month_absentees_percentage": f"{round(float(avg), 2)}%"
    })


# ---------------- RANGE ---------------- #

@app.route("/predict/range")
def predict_range():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    results = []
    current = start

    while current <= end:
        pred = smart_predict(current)

        results.append({
            "date": str(current.date()),
            "predicted_absentees_percentage": f"{round(float(pred), 2)}%"
        })

        current += timedelta(days=1)

    return jsonify(results)


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
