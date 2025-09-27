#!/usr/bin/env python3
# shebang line for linux/mac

import cv2 as cv
import numpy as np
from copy import deepcopy

image = cv.imread('Images/lake.jpg', cv.IMREAD_COLOR)

h, w, channels = image.shape
middle_with = round(w / 2)

factors = np.linspace(1, 0, 50)

for factor in factors:
    img = deepcopy(image)
    for y in range(0,h):
        for x in range(middle_with, w):
            bgr = img[y, x, :]
            bgr_dark = bgr * factor
            img[y, x, :] = bgr_dark
    
    cv.imshow('Imagem Escurecida', img)
    cv.waitKey(10)

cv.destroyAllWindows()
"""
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('darkening_effect.avi', fourcc, 30.0, (image.shape[1], image.shape[0]))

# Mostra a animação de escurecimento
for fator in np.linspace(1, 0.1, 30): # Escurece a imagem em 30 passos (desde o fator 1 até 0.1)
    darkimg = (image * fator).astype('uint8')
    out.write(darkimg)
    cv.imshow('Imagem Escurecida', darkimg)
    key = cv.waitKey(100)
    if key == 27:  # Se pressionar ESC, sai do loop
        break


out.release()

# Mantém a última imagem até pressionar uma tecla
cv.waitKey(0)
cv.destroyAllWindows()"""