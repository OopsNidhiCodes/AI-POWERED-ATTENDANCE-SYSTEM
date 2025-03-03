import cv2
import os

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Path where class image is stored
image_folder = "class_images"

# Get the latest image from the folder
def get_latest_image(folder):
    files = [f for f in os.listdir(folder) if f.endswith((".jpg", ".png"))]
    if not files:
        raise ValueError(f"❌ Error: No image found in {folder}!")
    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(folder, f)))
    return os.path.join(folder, latest_file)

# Detect faces and crop them
def detect_faces(image_path, output_folder):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("❌ Error: Could not read the image!")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        raise ValueError("❌ Error: No faces detected!")

    print(f"✅ Faces detected: {len(faces)}")

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    cropped_faces = []
    for i, (x, y, w, h) in enumerate(faces):
        face_crop = image[y:y+h, x:x+w]  # Crop the face region
        face_filename = os.path.join(output_folder, f"cropped_face_{i+1}.jpg")
        cv2.imwrite(face_filename, face_crop)
        cropped_faces.append(face_filename)
        print(f"✅ Cropped face saved: {face_filename}")

    return cropped_faces

# Main execution
if __name__ == "__main__":
    try:
        image_path = get_latest_image(image_folder)
        print(f"📸 Processing image: {image_path}")
        detect_faces(image_path, "cropped_faces")
    except Exception as e:
        print(str(e))
