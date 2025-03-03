import cv2
import os
from datetime import datetime

# Create a folder for storing images
save_path = "class_images"
os.makedirs(save_path, exist_ok=True)

# Open webcam (change 0 to the correct camera index if using an external camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("⚠️ Error: Could not access the camera!")

# Capture frame
ret, frame = cap.read()

if ret:
    # Generate filename based on timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_filename = f"class_{timestamp}.jpg"
    image_path = os.path.join(save_path, image_filename)

    # Save image
    cv2.imwrite(image_path, frame)
    print(f"✅ Image saved: {image_path}")

    # Show image preview
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(2000)  # Show for 2 seconds
    cv2.destroyAllWindows()

else:
    print("⚠️ Error: Failed to capture image!")

cap.release()
