import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

#kareler
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for point in landmarks.landmark:
                x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), -1)

#çubuklar
        for landmarks in results.multi_hand_landmarks:
            for connection in mp_hands.HAND_CONNECTIONS:
                x1, y1 = int(landmarks.landmark[connection[0]].x * frame.shape[1]), int(landmarks.landmark[connection[0]].y * frame.shape[0])
                x2, y2 = int(landmarks.landmark[connection[1]].x * frame.shape[1]), int(landmarks.landmark[connection[1]].y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Finger Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
