import os
import cv2
import numpy as np
from deepface import DeepFace
from insightface.app import FaceAnalysis

# Paths
cropped_faces_folder = "cropped_faces"
student_database_folder = "student_database"
attendance_file = "attendance.csv"
debug_cropped_faces_folder = "debug_cropped_faces"

# Create debug folder to save cropped images
os.makedirs(debug_cropped_faces_folder, exist_ok=True)

# Load the Face Analysis model (ArcFace)
face_analysis = FaceAnalysis(name='buffalo_l')
face_analysis.prepare(ctx_id=0, det_size=(640, 640))

# Get a list of student images
student_images = [os.path.join(student_database_folder, img) for img in os.listdir(student_database_folder) if img.endswith((".jpg", ".png"))]

# Face matching function
def match_faces():
    matched_students = set()

    for cropped_face in os.listdir(cropped_faces_folder):
        cropped_face_path = os.path.join(cropped_faces_folder, cropped_face)

        # Load the cropped image
        img = cv2.imread(cropped_face_path)
        if img is None:
            print(f"❌ Failed to read image: {cropped_face_path}")
            continue

        # Extract embeddings from the cropped face
        face_info = face_analysis.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not face_info:
            print(f"❌ No facial features found in {cropped_face_path}")
            continue
        
        # Extract the first detected face
        face_embedding = face_info[0].embedding

        # Print first 5 values of embedding for debugging
        print(f"📌 Extracted embedding for {cropped_face_path}: {face_embedding[:5]}")

        # Save detected face for verification
        bbox = face_info[0].bbox.astype(int)
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        cv2.imwrite(os.path.join(debug_cropped_faces_folder, f"{cropped_face}"), cropped_img)

        # Try to verify against student images
        matched = False
        for student_img in student_images:
            try:
                ref_img = cv2.imread(student_img)
                ref_info = face_analysis.get(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))

                if not ref_info:
                    print(f"❌ No facial features found in {student_img}")
                    continue

                ref_embedding = ref_info[0].embedding

                # Print first 5 values of reference embedding for debugging
                print(f"📌 Extracted embedding for {student_img}: {ref_embedding[:5]}")

                # Compute similarity score (Cosine Similarity)
                norm_face = np.linalg.norm(face_embedding)
                norm_ref = np.linalg.norm(ref_embedding)

                if norm_face == 0 or norm_ref == 0:
                    print(f"❌ Invalid embeddings (zero norm) for {cropped_face_path} or {student_img}")
                    continue

                similarity = np.dot(face_embedding, ref_embedding) / (norm_face * norm_ref)

                if similarity > 0.5:  # Adjusted threshold from 0.6 to 0.5
                    student_name = os.path.basename(student_img)
                    matched_students.add(student_name)
                    print(f"✅ Match Found: {student_name} (Similarity: {similarity:.2f})")
                    matched = True
                    break  # Stop checking once a match is found

            except Exception as e:
                print(f"❌ Error comparing {cropped_face_path} with {student_img}: {str(e)}")

        if not matched:
            print(f"❌ No match found for {cropped_face_path}")

    # Save matched students to attendance file
    with open(attendance_file, "w") as f:
        for student in matched_students:
            f.write(f"{student}, Present\n")
    
    # Print matched students
    print(f"📌 Matched students: {matched_students}")
    print(f"✅ Attendance file updated: {attendance_file}")

# Main execution
if __name__ == "__main__":
    try:
        match_faces()
    except Exception as e:
        print(str(e))
