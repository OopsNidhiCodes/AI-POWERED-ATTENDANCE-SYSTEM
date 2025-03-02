import cv2
import mediapipe as mp
import os

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.7)

# ORB Feature Detector
orb = cv2.ORB_create(nfeatures=1000)  # Increased feature points

# Eye landmark indices
LEFT_EYE = [33, 133, 160, 158, 144, 153]
RIGHT_EYE = [362, 263, 387, 385, 373, 380]

# Path where retina images are stored
SAVE_PATH = "retina_database"

# Load stored retina images
stored_images = {}
for file in os.listdir(SAVE_PATH):
    img_path = os.path.join(SAVE_PATH, file)
    stored_img = cv2.imread(img_path, 0)  # Load in grayscale
    if stored_img is not None:
        stored_images[file] = stored_img

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
                        gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

                        # Detect features in the live eye scan
                        kp_live, des_live = orb.detectAndCompute(gray_eye, None)
                        print(f"Live scan - {name} eye: {len(kp_live) if kp_live else 0} keypoints detected.")

                        best_match = None
                        best_score = float('-inf')

                        for filename, stored_img in stored_images.items():
                            kp_stored, des_stored = orb.detectAndCompute(stored_img, None)
                            print(f"Stored image - {filename}: {len(kp_stored) if kp_stored else 0} keypoints detected.")

                            if des_live is not None and des_stored is not None:
                                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                                matches = bf.match(des_live, des_stored)
                                score = len(matches)

                                if score > best_score:
                                    best_score = score
                                    best_match = filename

                        if best_match and best_score > 15:  # Increased threshold
                            cv2.putText(frame, f"Matched: {best_match}", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "No Match Found", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    cv2.imshow("Retina Matching Debug", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()