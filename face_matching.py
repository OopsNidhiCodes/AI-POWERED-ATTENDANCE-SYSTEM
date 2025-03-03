import os
import cv2
from deepface import DeepFace

# Paths
cropped_faces_folder = "cropped_faces"
student_database_folder = "student_database"

# Get a list of student images
student_images = [os.path.join(student_database_folder, img) for img in os.listdir(student_database_folder) if img.endswith((".jpg", ".png"))]

# Process each detected face
for cropped_face in os.listdir(cropped_faces_folder):
    cropped_face_path = os.path.join(cropped_faces_folder, cropped_face)

    # Load the cropped image to check if it's valid
    img = cv2.imread(cropped_face_path)
    if img is None:
        print(f"❌ Failed to read image: {cropped_face_path}")
        continue

    # Try to verify against student images
    matched = False
    for student_img in student_images:
        try:
            result = DeepFace.verify(img1_path=cropped_face_path, img2_path=student_img)
            if result["verified"]:
                print(f"✅ Match Found: {student_img}")
                matched = True
                with open("attendance.csv", "a") as f:
                    f.write(f"{os.path.basename(student_img)}, Present\n")
                break  # Stop checking once a match is found
        except Exception as e:
            print(f"❌ Error comparing {cropped_face_path} with {student_img}: {str(e)}")

    if not matched:
        print(f"❌ No match found for {cropped_face_path}")
