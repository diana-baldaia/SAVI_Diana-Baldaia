import cv2 as cv
import numpy as np

# H = altura, número de linhas --> L
# W = largura, número de colunas --> C

# Imagem school
school_path = '/home/baldaia/Desktop/SAVI_Diana-Baldaia/Parte02/Images/school.png'
img_school = cv.imread(school_path, cv.IMREAD_COLOR)
H, W, channels = img_school.shape

# Imagem beach
beach_path = '/home/baldaia/Desktop/SAVI_Diana-Baldaia/Parte02/Images/beach.png'
img_beach = cv.imread(beach_path, cv.IMREAD_COLOR)
L, C, channels = img_beach.shape

# Imagem wally - template
templatepath = '/home/baldaia/Desktop/SAVI_Diana-Baldaia/Parte02/Images/wally.png'
template = cv.imread(templatepath, cv.IMREAD_COLOR)
h, w, channels = template.shape



# --- School ---
result_school = cv.matchTemplate(img_school, template, cv.TM_CCORR_NORMED)
print ('result_school =', str(result_school))
print('result_school type =', str(result_school.dtype))

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result_school)
print('max_loc =', str(max_loc))

# Desenha um retângulo à volta do Wally
top_left = max_loc
bottom_right = (top_left[0] + h, top_left[1] + w)
cv.rectangle(img_school, top_left, bottom_right, (255, 0, 0), 3)


cv.imshow('Original', img_school)
cv.imshow('Template', template)
cv.imshow('Result', result_school)
cv.waitKey(0)



# --- Beach ---
result_beach = cv.matchTemplate(img_beach, template, cv.TM_CCORR_NORMED)
print ('result_beach =', str(result_beach))
print('result_beach type =', str(result_beach.dtype))

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result_beach)
print('max_loc =', str(max_loc))

# Desenha um retângulo à volta do Wally
top_left = max_loc
bottom_right = (top_left[0] + h, top_left[1] + w)
cv.rectangle(img_beach, top_left, bottom_right, (255, 0, 0), 3)


cv.imshow('Original', img_beach)
cv.imshow('Template', template)
cv.imshow('Result', result_beach)
cv.waitKey(0)


cv.destroyAllWindows()