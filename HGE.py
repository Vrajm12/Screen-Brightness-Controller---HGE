import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc # type: ignore

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Flip the image for a mirror effect
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of thumb and index finger tips
            h, w, _ = img.shape
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_coords = (int(index_tip.x * w), int(index_tip.y * h))

            # Draw circles
            cv2.circle(img, thumb_coords, 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, index_coords, 10, (0, 255, 0), cv2.FILLED)

            # Calculate the distance
            distance = hypot(index_coords[0] - thumb_coords[0], index_coords[1] - thumb_coords[1])

            # Adjust brightness based on distance
            if distance < 50:
                sbc.set_brightness(50)
            else:
                sbc.set_brightness(100)

    # Display the image
    cv2.imshow("Hand Gesture Control", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
