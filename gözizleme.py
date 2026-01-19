import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

LEFT_EYE_LANDMARKS = [33, 133, 159, 145]   
RIGHT_EYE_LANDMARKS = [362, 263, 386, 374] 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape

            def get_eye_center(landmarks, indices):
                x = int(sum(face_landmarks.landmark[i].x for i in indices) / len(indices) * iw)
                y = int(sum(face_landmarks.landmark[i].y for i in indices) / len(indices) * ih)
                return (x, y)

            left_eye_center = get_eye_center(face_landmarks.landmark, LEFT_EYE_LANDMARKS)
            right_eye_center = get_eye_center(face_landmarks.landmark, RIGHT_EYE_LANDMARKS)

            cv2.circle(frame, left_eye_center, 5, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_center, 5, (0, 255, 0), -1)

            cv2.line(frame, left_eye_center, right_eye_center, (255, 0, 0), 2)

            gaze_x = (left_eye_center[0] + right_eye_center[0]) // 2

            if gaze_x < iw // 2 - iw // 10:
                direction = "Sola bak"
            elif gaze_x > iw // 2 + iw // 10:
                direction = "Saga bak "
            else:
                direction = "Ortaya bak"

            cv2.putText(frame, direction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
