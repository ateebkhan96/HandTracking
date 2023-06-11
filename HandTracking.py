# Importing the required libraries
import cv2
import mediapipe as mp
import time  # Time module for fps

cap = cv2.VideoCapture(0)  # getting the video from webcam

# Using the mediapipe library to get the hand landmarks
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0  # previous time
currTime = 0  # current time

while True:

    ret, frame = cap.read()  # reading the frame

    # converting the bgr to rgb
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # using the mediapipe hands module to get the detection results
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:  # Checking if hand landmarks are detected
        # Perform operations for each detected hand
        for handLms in results.multi_hand_landmarks:
            for ids, lm in enumerate(handLms.landmark):  # getting the id's and landmarks

                h, w, c = frame.shape  # getting heigth, width and color channel
                cX, cY = int(lm.x * w), int(lm.y * h)  # converting the x and y points to pixel values
                print(ids, cX, cY)
                # Uncomment if you want to change the markings on indivitual points
                # if ids == 1:
                #     cv2.circle(frame,(cX,cY),15,(0,0,255),-1)
            # drawing the landmarks
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    # fps counter
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
