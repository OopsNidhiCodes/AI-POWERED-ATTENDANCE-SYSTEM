from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from capture_class_images import capture_image
from face_detection import detect_faces
from face_matching import match_faces

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Dummy login credentials
USERNAME = "admin"
PASSWORD = "admin123"

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid Credentials!", "danger")

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("dashboard.html")

@app.route("/capture", methods=["POST"])
def capture():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    image_path, error = capture_image()
    if image_path:
        flash("Image Captured Successfully!", "success")
        return redirect(url_for("process", image=image_path))
    
    flash(error, "danger")
    return redirect(url_for("dashboard"))

@app.route("/upload", methods=["POST"])
def upload():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    file = request.files["file"]
    if file:
        image_path = os.path.join("class_images", file.filename)
        file.save(image_path)
        flash("Image Uploaded Successfully!", "success")
        return redirect(url_for("process", image=image_path))

    flash("No file selected!", "danger")
    return redirect(url_for("dashboard"))

@app.route("/process/<path:image>")
def process(image):
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    faces, error = detect_faces()
    if faces:
        matched_faces = match_faces()
        return render_template("attendance.html", faces=matched_faces)
    
    flash(error, "danger")
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
