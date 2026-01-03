from flask import Flask, request
from resume_parser import extract_text_from_pdf
from matcher import match_resume
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    score = None

    if request.method == "POST":
        resume = request.files["resume"]
        job_desc = request.form["job_desc"]

        resume_path = os.path.join(UPLOAD_FOLDER, resume.filename)
        resume.save(resume_path)

        resume_text = extract_text_from_pdf(resume_path)
        score = match_resume(resume_text, job_desc)

    return f"""
    <h2>AI-Based Resume Screening System</h2>
    <form method="post" enctype="multipart/form-data">
        <p>Upload Resume (PDF)</p>
        <input type="file" name="resume" required><br><br>
        <p>Paste Job Description</p>
        <textarea name="job_desc" rows="6" cols="60" required></textarea><br><br>
        <button type="submit">Evaluate</button>
    </form>
    <h3>Match Score: {score if score else ""}</h3>
    """

if __name__ == "__main__":
    app.run(debug=True)
