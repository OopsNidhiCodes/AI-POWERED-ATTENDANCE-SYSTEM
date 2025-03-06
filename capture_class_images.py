import cv2
import os
from datetime import datetime

# Define function to capture class image
def capture_image():
    save_path = "class_images"
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("⚠️ Error: Could not access the camera!")
        return None

    ret, frame = cap.read()
    cap.release()  # Release the camera after capturing

    if ret:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = f"class_{timestamp}.jpg"
        image_path = os.path.join(save_path, image_filename)
        cv2.imwrite(image_path, frame)
        print(f"✅ Image saved: {image_path}")
        return image_path, None  # Return the path of the captured image
    else:
        print("⚠️ Error: Failed to capture image!")
        return None
capture_image()
