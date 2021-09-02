import cv2
import mediapipe as mp
from time import sleep
import numpy as np
import autopy
import pynput

wCam, hCam = 1280, 720
wScr, hScr = autopy.screen.size()

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

mouse = pynput.mouse.Controller()

def findNodeDistance(imgHeight, imgWidth, landmarks, index1, index2):
    x1 = int(landmarks[index1].x*imgWidth)
    y1 = int(landmarks[index1].y*imgHeight)
    z1 = int(landmarks[index1].z*imgWidth)
    x2 = int(landmarks[index2].x*imgWidth)
    y2 = int(landmarks[index2].y*imgHeight)
    z2 = int(landmarks[index2].z*imgWidth)

    dis = ((x1-x2)**2.0+(y1-y2)**2.0)**0.5
    z_dis = abs(z1-z2)
    return dis, z_dis

with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue    

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)    

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                #cx, cy = int(hand_landmarks.landmark[8].x*wCam), int(hand_landmarks.landmark[8].y*hCam)
                targetX, targetY = int(hand_landmarks.landmark[8].x*wScr), int(hand_landmarks.landmark[8].y*hScr)
                mouse.position = (targetX, targetY)

                xy_dis_8_12, z_dis_8_12 = findNodeDistance(hCam, wCam, hand_landmarks.landmark, 8, 12)
                xy_dis_12_16, z_dis_12_16 = findNodeDistance(hCam, wCam, hand_landmarks.landmark, 12, 16)

                if xy_dis_8_12 < 40 and z_dis_8_12 < 20:
                    mouse.click(pynput.mouse.Button.left)
                    sleep(0.3)
                if xy_dis_12_16 < 40 and z_dis_12_16 < 20:
                    mouse.click(pynput.mouse.Button.left, 2)
                    sleep(0.3)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()