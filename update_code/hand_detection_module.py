# hand_detection_module.py

import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        detected_hands_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for point in hand_landmarks.landmark:
                    x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[2])
                    landmarks.append((x, y, z))
                detected_hands_landmarks.append(landmarks)

        return detected_hands_landmarks

    def fingers_up(self, landmarks):
        fingers = []

        # Thumb
        if landmarks[self.tipIds[0]][1] < landmarks[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if landmarks[self.tipIds[id]][2] < landmarks[self.tipIds[id]-2][2]:
                fingers.append(1)  # open
            else:
                fingers.append(0)  # closed

        return fingers

    def main(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            detected_hands = self.find_hands(frame)

            for landmarks in detected_hands:
                for x, y, z in landmarks:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw hand connections
                connections = self.mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    x1, y1, _ = landmarks[connection[0]]
                    x2, y2, _ = landmarks[connection[1]]
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Example: Use fingers_up method
                fingers_status = self.fingers_up(landmarks)
                print(f"Fingers Status: {fingers_status}")

            cv2.imshow('Hand Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    hand_detector = HandDetector()
    hand_detector.main()
