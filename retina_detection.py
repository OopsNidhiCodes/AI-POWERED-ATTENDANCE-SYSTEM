import cv2
import mediapipe as mp
import os

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7)

# Eye landmark indices
LEFT_EYE = [33, 133, 160, 158, 144, 153]
RIGHT_EYE = [362, 263, 387, 385, 373, 380]

# Create a folder to save images if it doesn't exist
SAVE_PATH = "retina_database"
os.makedirs(SAVE_PATH, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get eye bounding boxes
            for eye, name in [(LEFT_EYE, "left"), (RIGHT_EYE, "right")]:
                eye_pts = [(int(face_landmarks.landmark[idx].x * w),
                            int(face_landmarks.landmark[idx].y * h)) for idx in eye]

                if eye_pts:
                    x_min = min(p[0] for p in eye_pts)
                    y_min = min(p[1] for p in eye_pts)
                    x_max = max(p[0] for p in eye_pts)
                    y_max = max(p[1] for p in eye_pts)

                    eye_img = frame[y_min:y_max, x_min:x_max]

                    if eye_img.size > 0:
                        cv2.imwrite(f"{SAVE_PATH}/student_{name}.jpg", eye_img)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    cv2.imshow("Retina Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()