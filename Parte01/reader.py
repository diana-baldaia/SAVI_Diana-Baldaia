import cv2 as cv
import numpy as np

cap = cv.VideoCapture('darkening_effect.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv.imshow('Frame', frame)
    key = cv.waitKey(30)
    if key == 27:  # Se pressionar ESC, sai do loop
        break

cap.release()
cv.destroyAllWindows()