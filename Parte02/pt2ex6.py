import cv2 as cv
import numpy as np

# Imagem finding_wally
filepath = '/home/baldaia/Desktop/SAVI_Diana-Baldaia/Parte02/Images/finding_wally.png'
img = cv.imread(filepath, cv.IMREAD_COLOR)
H, W, channels = img.shape

# Imagem wally --> template
templatepath = '/home/baldaia/Desktop/SAVI_Diana-Baldaia/Parte02/Images/wally.png'
template = cv.imread(templatepath, cv.IMREAD_COLOR)
h, w, channels = template.shape

# Aplica a template matching - encontra o wally na imagem
result = cv.matchTemplate(img, template, cv.TM_CCORR_NORMED)
print ('result =', str(result))
print('result type =', str(result.dtype))

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
print('max_loc =', str(max_loc))



# Colocar a imagem a cinza exceto o wally
mask = np.zeros((H, W), dtype=np.uint8)  # Máscara inicial a preto
mask[max_loc[1]:max_loc[1]+w, max_loc[0]:max_loc[0]+h] = 1  # Região do wally a branco
mask_inv = 1 - mask  # Inverso da máscara, wally a preto, resto a branco
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Imagem a cinza
img_gray_bgr = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)  # Converter para BGR para poder misturar
img_result = img * mask[:, :, np.newaxis] + img_gray_bgr * mask_inv[:, :, np.newaxis]   # Mistura as duas imagens



cv.imshow('Original', img)
cv.imshow('Template', template)
cv.imshow('Result', result)
cv.imshow('Wally Highlighted', img_result)

cv.waitKey(0)
cv.destroyAllWindows()