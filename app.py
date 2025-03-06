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

# Ensure class_images directory exists
if not os.path.exists("class_images"):
    os.makedirs("class_images")

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

    if "file" not in request.files:
        flash("No file part", "danger")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file", "danger")
        return redirect(request.url)

    if file:
        filename = file.filename
        image_path = os.path.join("class_images", filename)
        try:
            file.save(image_path)
            flash("Image Uploaded Successfully!", "success")
            return redirect(url_for("process", image=image_path))
        except Exception as e:
            flash(f"Error saving file: {e}", "danger")
            return redirect(url_for("dashboard"))

    flash("No file selected!", "danger")
    return redirect(url_for("dashboard"))

@app.route("/process")
def process():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        faces, error = detect_faces()
        if faces:
            matched_faces = match_faces()
            print("Matched Faces:", matched_faces)
            return render_template("attendance.html", faces=matched_faces)
        else:
            flash(error or "No faces detected.", "warning") # Use warning if detect_faces return empty error
            return redirect(url_for("dashboard"))
    except Exception as e:
        flash(f"Error processing image: {e}", "danger")
        return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)