# prompt_mistral.py

def get_mistral_prompt(student_id, student_name, advisor_name, courses, total_credits, query):
    """
    Build a tightly formatted prompt with clear delimiters and headers.
    """

    lines = []

    # 1) Clear instruction
    lines.append("You are a personal RAG assistant for Rowan University.")
    lines.append("Below is a student’s schedule. Give a detailed answer pls, include any necessary information that may have not been asked for but maybe useful for the student\n")

    # 2) Context delimiters
    lines.append("<context>")
    lines.append(f"Student ID   : {student_id}")
    lines.append(f"Student Name : {student_name}")
    lines.append(f"Advisor      : {advisor_name}\n")

    lines.append("Courses:")
    for code, name, term, inst, days, time_, building, room, cred in courses:
        lines.append(f"- Course Code     : {code}")
        lines.append(f"  Course Name     : {name}")
        lines.append(f"  Term            : {term}")
        lines.append(f"  Instructor Name : {inst}")
        lines.append(f"  Days            : {days}")
        lines.append(f"  Time            : {time_}")
        lines.append(f"  Building        : {building}")
        lines.append(f"  Room Number     : {room}")
        lines.append(f"  Credits         : {cred}\n")

    lines.append(f"Total Credits      : {total_credits}")
    lines.append("</context>\n")

    # 3) User’s question
    lines.append(f"Question: {query}")
    lines.append("\nAnswer:")

    # Join with newlines
    return "\n".join(lines)
