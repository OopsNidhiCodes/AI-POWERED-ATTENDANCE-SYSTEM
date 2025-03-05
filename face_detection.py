import cv2
import os
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

# Path where class image is stored
image_folder = "class_images"
output_folder = "cropped_faces"

# Get the latest image from the folder
def get_latest_image(folder):
    files = [f for f in os.listdir(folder) if f.endswith((".jpg", ".png"))]
    if not files:
        raise ValueError(f"❌ Error: No image found in {folder}!")
    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(folder, f)))
    return os.path.join(folder, latest_file)

# Detect faces and crop them
def detect_faces():
    image_path = get_latest_image(image_folder)
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("❌ Error: Could not read the image!")

    # Detect faces using MTCNN
    faces = detector.detect_faces(image)
    
    if not faces:
        raise ValueError("❌ Error: No faces detected!")

    print(f"✅ Faces detected: {len(faces)}")

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    cropped_faces = []
    for i, face in enumerate(faces):
        x, y, w, h = face['box']

        # Expand bounding box slightly to avoid cropping too much
        x, y, w, h = max(0, x-10), max(0, y-10), w+20, h+20

        face_crop = image[y:y+h, x:x+w]  # Crop the face region
        face_filename = os.path.join(output_folder, f"cropped_face_{i+1}.jpg")
        cv2.imwrite(face_filename, face_crop)
        cropped_faces.append(face_filename)
        print(f"✅ Cropped face saved: {face_filename}")

    return cropped_faces

# Main execution (for testing)
if __name__ == "__main__":
    try:
        detect_faces()
    except Exception as e:
        print(str(e))
