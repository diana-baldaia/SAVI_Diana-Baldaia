import cv2 as cv
import numpy as np

img = cv.imread('Images/praia.png')
cv.imshow('Original', img)
cv.waitKey(0)
h, w, channels = img.shape


# --- Decidir para cada pixel se é céu (1) ou não (0) ---

segmentation_mask = np.ndarray((h, w,), dtype=np.uint8)  # Máscara binária (0 ou 1) para céu

for y in range(h): # linha
    for x in range(w): # colunas
        b, g, r = img[y, x, :]

        if b > g and b > r:
            segmentation_mask[y, x] = 255  # Céu
        elif r > b and r > g:
            segmentation_mask[y, x] = 125  # Areia
        else:
            segmentation_mask[y, x] = 0  # Mar


cv.imshow('Segmentacao', segmentation_mask)  # Mostrar a máscara de segmentação, passando o 1 para 255 (branco)
cv.waitKey(0)
cv.destroyAllWindows()