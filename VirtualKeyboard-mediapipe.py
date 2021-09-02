import cv2
import mediapipe as mp
from time import sleep
import numpy as np
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

keyboard = Controller()

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

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

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([200 + 100 * j + 50, 100 * i + 50], key))

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

        image = drawAll(image, buttonList)    

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                img_h, img_w, img_c = image.shape
                cx, cy = int(hand_landmarks.landmark[8].x*img_w), int(hand_landmarks.landmark[8].y*img_h)

                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size

                    if x < cx <x+w and y < cy <y+h:
                        cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                        cv2.putText(image, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        xy_dis_8_12, z_dis_8_12 = findNodeDistance(img_h, img_w, hand_landmarks.landmark, 8, 12)

                        if xy_dis_8_12 < 40 and z_dis_8_12 < 40:
                            keyboard.press(button.text)
                            cv2.rectangle(image, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(image, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                            finalText += button.text
                            finalText = finalText[-12:]
                            sleep(0.3)

        cv2.rectangle(image, (250, 350), (900, 450), (255, 0, 255), cv2.FILLED)
        cv2.putText(image, finalText, (260, 430),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()