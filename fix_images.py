import cv2
import os

cropped_faces_folder = "cropped_faces"
fixed_faces_folder = "fixed_faces"

# Create a new folder for properly formatted images
os.makedirs(fixed_faces_folder, exist_ok=True)

for cropped_face in os.listdir(cropped_faces_folder):
    cropped_face_path = os.path.join(cropped_faces_folder, cropped_face)
    fixed_face_path = os.path.join(fixed_faces_folder, cropped_face)

    # Load and re-save the image in proper format
    img = cv2.imread(cropped_face_path)
    if img is None:
        print(f"❌ Failed to read image: {cropped_face_path}")
        continue

    cv2.imwrite(fixed_face_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"✅ Image fixed and saved: {fixed_face_path}")
