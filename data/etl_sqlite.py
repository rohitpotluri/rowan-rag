import sqlite3
import pandas as pd
import hashlib
import os

# Paths to your data files
DATA_DIR = os.path.join(os.path.dirname(__file__))
CSV_PATH = os.path.join(DATA_DIR, "student_schedule.csv")
DB_PATH  = os.path.join(DATA_DIR, "schedule.db")

# 1) Load the CSV
df = pd.read_csv(CSV_PATH)

# 2) Build Students table (for login)
students = (
    df[["Student ID", "Student Name", "Advisor Name"]]
    .drop_duplicates()
    .rename(columns={
        "Student ID": "student_id",
        "Student Name": "student_name",
        "Advisor Name": "advisor_name",
    })
)
# Hash each “password<SID>”
students["password_hash"] = students["student_id"].apply(
    lambda sid: hashlib.sha256(f"password{sid}".encode()).hexdigest()
)

# 3) Build StudentCourses table (denormalized raw data)
student_courses = (
    df.rename(columns={
        "Student ID":       "student_id",
        "Student Name":     "student_name",
        "Advisor Name":     "advisor_name",
        "Course Code":      "course_code",
        "Course Name":      "course_name",
        "Instructor Name":  "instructor_name",
        "Days":             "days",
        "Time":             "time",
        "Building":         "building",
        "Room Number":      "room_number",
        "Credits":          "credits",
        "Term":             "term"
    })
    # Ensure each original row is preserved for that student
    .drop_duplicates()
)

# 4) Write to SQLite
conn = sqlite3.connect(DB_PATH)

# Replace existing tables
students.to_sql("Students",          conn, if_exists="replace", index=False)
student_courses.to_sql("StudentCourses", conn, if_exists="replace", index=False)

conn.close()

print(f"Data loaded into {DB_PATH}")
