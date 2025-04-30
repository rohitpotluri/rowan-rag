# db_utils.py
import sqlite3
import hashlib
import os

# Point to your data folder’s DB
DB_PATH = os.path.join("data", "schedule.db")

def get_password_hash(student_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT password_hash FROM Students WHERE student_id = ?",
        (student_id,)
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def get_student_context(student_id):
    """
    Returns:
      student_name  (str),
      advisor_name  (str),
      courses       (List[Tuple(course_code, course_name, term, instructor_name, days, time, building, room_number, credits)]),
      total_credits (int)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1) Student name & advisor
    cur.execute(
        "SELECT student_name, advisor_name FROM Students WHERE student_id = ?",
        (student_id,)
    )
    student_name, advisor_name = cur.fetchone()

    # 2) All enrolled courses’ details from the denormalized table
    cur.execute("""
        SELECT
            course_code,
            course_name,
            term,
            instructor_name,
            days,
            time,
            building,
            room_number,
            credits
        FROM StudentCourses
        WHERE student_id = ?
    """, (student_id,))
    courses = cur.fetchall()

    conn.close()

    # 3) Compute total credits
    total_credits = sum(row[-1] for row in courses)

    return student_name, advisor_name, courses, total_credits

def verify_credentials(student_id, password):
    stored = get_password_hash(student_id)
    if not stored:
        return False
    # Hash the entered password the same way ETL did
    candidate_hash = hashlib.sha256(password.encode()).hexdigest()
    return candidate_hash == stored
