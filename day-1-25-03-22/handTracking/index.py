from turtle import color
import cv2
import mediapipe

from datetime import datetime

cam = cv2.VideoCapture(0)
hand = mediapipe.solutions.hands
hands = hand.Hands()
draw = mediapipe.solutions.drawing_utils

point = draw.DrawingSpec(color=(0,0,255), thickness=5)
line = draw.DrawingSpec(color=(0,255,0), thickness=7)


while True:
    ret, img = cam.read()

    if ret:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(img)
        finger = res.multi_hand_landmarks

        if finger:
            for fingers in finger:
                draw.draw_landmarks(img, fingers, hand.HAND_CONNECTIONS, point,line)
                #print("Hand dectected! " + datetime.now().strftime('%H:%M:%S')

                for i, j in enumerate(fingers.landmark):
                    print(i, j.x, j.y)

        cv2.imshow("Hanad Tracking", img)

    if cv2.waitKey(1) == ord('q'):
        break