import cv2 as cv
import numpy as np

filepath = '/home/baldaia/Desktop/SAVI_Diana-Baldaia/Parte02/Images/finding_wally.png'
img = cv.imread(filepath, cv.IMREAD_COLOR)
H, W, channels = img.shape

templatepath = '/home/baldaia/Desktop/SAVI_Diana-Baldaia/Parte02/Images/wally.png'
template = cv.imread(templatepath, cv.IMREAD_COLOR)
h, w, channels = template.shape


result = cv.matchTemplate(img, template, cv.TM_CCORR_NORMED)
print ('result =', str(result))
print('result type =', str(result.dtype))

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
print('max_loc =', str(max_loc))

# Desenha um retângulo à volta do Wally
top_left = max_loc
bottom_right = (top_left[0] + h, top_left[1] + w)
cv.rectangle(img, top_left, bottom_right, (255, 0, 0), 3)


cv.imshow('Original', img)
cv.imshow('Template', template)
cv.imshow('Result', result)

cv.waitKey(0)
cv.destroyAllWindows()