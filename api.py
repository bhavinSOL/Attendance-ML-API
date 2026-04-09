from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import base64
from github import Github, GithubException

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODEL ---------------- #

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- LOAD ADMIN PASSWORD FROM CSV ---------------- #

import os
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'public')
PASSWORDS_CSV_PATH = os.path.join(PUBLIC_DIR, 'passwords.csv')
PASSWORDS_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "bhavinSOL/TATA-Attendance/refs/heads/main/public/passwords.csv"
)

def load_admin_password():
    """Load admin password from local CSV or remote URL"""
    try:
        # Try remote first
        pwd_df = pd.read_csv(PASSWORDS_CSV_URL)
        admin_row = pwd_df[pwd_df['username'] == 'admin']
        if len(admin_row) > 0:
            return str(admin_row.iloc[0]['password'])
    except:
        pass

    # Fallback to local
    try:
        pwd_df = pd.read_csv(PASSWORDS_CSV_PATH)
        admin_row = pwd_df[pwd_df['username'] == 'admin']
        if len(admin_row) > 0:
            return str(admin_row.iloc[0]['password'])
    except:
        pass

    return "admin123"  # Default fallback

ADMIN_PASSWORD = load_admin_password()
print(f"✅ Loaded admin password from CSV")

# ---------------- LOAD CALENDAR ---------------- #

import os
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'public')

# Primary source: online CSV from GitHub (kept in sync with repo)
CALENDAR_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "bhavinSOL/TATA-Attendance/refs/heads/main/public/2026_calander.csv"
)

try:
    calendar_df = pd.read_csv(CALENDAR_CSV_URL)
    print("✅ Loaded calendar from remote URL")
except Exception as e:
    print(f"⚠️ Remote calendar load failed ({e}); falling back to local file")
    calendar_df = pd.read_csv(os.path.join(PUBLIC_DIR, "2026_calander.csv"))

calendar_df["date"] = pd.to_datetime(calendar_df["date"], dayfirst=True)

# ---------------- LOAD ATTENDANCE HISTORY ---------------- #

ATTENDANCE_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "bhavinSOL/TATA-Attendance/refs/heads/main/public/attendance.csv"
)

try:
    attendance_df = pd.read_csv(ATTENDANCE_CSV_URL)
    print("✅ Loaded attendance from remote URL")
except Exception as e:
    print(f"⚠️ Remote attendance load failed ({e}); falling back to local file")
    attendance_df = pd.read_csv(os.path.join(PUBLIC_DIR, "attendance.csv"))
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

    return jsonify({
        "date": date,
        "predicted_absentees_percentage": f"{round(float(model_prediction), 2)}%"
    })


# ---------------- HELPER: Simple model prediction for all endpoints ---------------- #

def smart_predict(date):
    """
    Simple model prediction without hybrid logic.
    Just returns the pure model prediction based on features.
    """
    features = build_features(date)
    model_pred = model.predict(features)[0]
    return model_pred


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


# -------- GITHUB API ENDPOINTS --------

@app.route("/github/update-csv", methods=["POST"])
def update_csv_github():
    """
    Update CSV file in GitHub repository.
    Expected JSON body:
    {
        "filename": "2026_calander.csv" or "attendance.csv",
        "content": "CSV content as string",
        "message": "Commit message"
    }
    """
    try:
        data = request.get_json()
        filename = data.get("filename")
        content = data.get("content")
        message = data.get("message", f"Update {filename}")

        if not filename or not content:
            return jsonify({"error": "Missing filename or content"}), 400

        # Get GitHub credentials from environment
        github_token = os.environ.get("GITHUB_TOKEN")
        github_repo = os.environ.get("GITHUB_REPO", "bhavinSOL/TATA-Attendance")

        if not github_token:
            return jsonify({
                "error": "GitHub token not configured. Please set GITHUB_TOKEN environment variable.",
                "fallback": True
            }), 500

        # Initialize GitHub client
        g = Github(github_token)
        repo = g.get_repo(github_repo)

        # Path in repository
        file_path = f"public/{filename}"

        try:
            # Try to get existing file to update it
            contents = repo.get_contents(file_path)
            repo.update_file(
                path=file_path,
                message=message,
                content=content,
                sha=contents.sha
            )
            return jsonify({
                "success": True,
                "message": f"{filename} updated successfully in GitHub"
            }), 200
        except GithubException as e:
            if e.status == 404:
                # File doesn't exist, create it
                repo.create_file(
                    path=file_path,
                    message=message,
                    content=content
                )
                return jsonify({
                    "success": True,
                    "message": f"{filename} created successfully in GitHub"
                }), 201
            else:
                raise e

    except GithubException as e:
        return jsonify({
            "error": f"GitHub API error: {str(e)}",
            "fallback": True
        }), 500
    except Exception as e:
        print(f"Error updating CSV on GitHub: {e}")
        return jsonify({
            "error": str(e),
            "fallback": True
        }), 500


# -------- PASSWORD CHANGE ENDPOINT --------

@app.route("/auth/verify-password", methods=["POST"])
def verify_password():
    """
    Verify current admin password.
    Expected JSON body:
    {
        "currentPassword": "password_to_verify"
    }
    """
    try:
        data = request.get_json()
        current_password = data.get("currentPassword", "")

        if current_password == ADMIN_PASSWORD:
            return jsonify({
                "success": True,
                "message": "Password verified successfully"
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Incorrect password"
            }), 401

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/auth/change-password", methods=["POST"])
def change_password():
    """
    Change admin password and save to passwords.csv
    Expected JSON body:
    {
        "oldPassword": "current_password",
        "newPassword": "new_password"
    }
    """
    global ADMIN_PASSWORD

    try:
        data = request.get_json()
        old_password = data.get("oldPassword", "")
        new_password = data.get("newPassword", "")

        if not old_password or not new_password:
            return jsonify({
                "error": "Missing old or new password"
            }), 400

        # Verify old password
        if old_password != ADMIN_PASSWORD:
            return jsonify({
                "success": False,
                "error": "Old password is incorrect"
            }), 401

        # Update in-memory password
        ADMIN_PASSWORD = new_password

        # Update passwords.csv
        pwd_df = pd.DataFrame({'username': ['admin'], 'password': [new_password]})
        pwd_df.to_csv(PASSWORDS_CSV_PATH, index=False)

        # Also update on GitHub
        try:
            github_token = os.environ.get("GITHUB_TOKEN")
            github_repo = os.environ.get("GITHUB_REPO", "bhavinSOL/TATA-Attendance")

            if github_token:
                g = Github(github_token)
                repo = g.get_repo(github_repo)
                file_path = "public/passwords.csv"

                try:
                    contents = repo.get_contents(file_path)
                    repo.update_file(
                        path=file_path,
                        message="Update admin password",
                        content=pwd_df.to_csv(index=False),
                        sha=contents.sha
                    )
                except GithubException as e:
                    if e.status == 404:
                        repo.create_file(
                            path=file_path,
                            message="Create passwords file",
                            content=pwd_df.to_csv(index=False)
                        )
        except Exception as e:
            print(f"⚠️ GitHub update failed: {e}")

        return jsonify({
            "success": True,
            "message": "Password changed successfully"
        }), 200

    except Exception as e:
        print(f"Error changing password: {e}")
        return jsonify({
            "error": str(e)
        }), 500


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    # Render provides the port via the PORT environment variable.
    # Locally, this falls back to 8000.
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
