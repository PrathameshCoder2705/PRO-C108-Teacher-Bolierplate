import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.8    )

def drawHandLandmarks(image, hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

while True:
    success, image = cap.read()
    
    # detect the hands landmark 
    results = hands.process(image)
    
    # Get landmark position from the processed results  
    hand_landmarks = results.multi_hand_landmarks
    
    # Call the function to draw landmarks 
    drawHandLandmarks(image, hand_landmarks)
    
    cv.imshow("Media Controller", image)

    key = cv.waitKey(1)
    if key == 32:
        break

cv.destroyAllWindows()
