import cv2
import mediapipe as mp

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize drawing utils
mp_drawing = mp.solutions.drawing_utils # draws all hand's landmarks pts on the o/p img 


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the i/p frame to dtect  hands and their landmarks
    results = hands.process(frame_rgb) 

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:#olds a list of detected hand landmarks if any hands are detected in the frame.
        for hand_landmarks in results.multi_hand_landmarks: #iterates over each detected hand's landmarks.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # defines conn b/w landmarks like like to form skeltion of hand 

    cv2.imshow('Hand Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
