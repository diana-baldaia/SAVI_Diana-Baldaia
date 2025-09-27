import cv2 as cv
import numpy as np

def main():
    img = cv.imread('lake.jpg', cv.IMREAD_COLOR)

    print(img.dtype) # Data type of the image array
    print(type(img)) # Type of the image object
    print('shape = ', str(img.shape)) # Shape of the image array

    color = img[5, 5] # Accessing the color of the pixel at (5, 5)
    print('Cor do pixel (5,5): ', color)

    img[0:50, 0:70] = 0 # Setting a region of the image to black

    B, G, R = cv.split(img) # Splitting the image into its color channels
    mask = B > 220 # Creating a mask for high blue values
    print('Mask dtype 1: ' + str(mask.dtype))
    mask = mask.astype('uint8') * 255 # Converting boolean mask to uint8 (tudo o que tenha azul acima de 220 passa a branco - 255)
    print('Mask dtype 2: ' + str(mask.dtype))

    azul_mask = np.zeros_like(img) # Creating a black image for blue regions same size as original
    azul_mask[mask == 255] = [255, 0, 0] # Setting blue regions to pure blue in BGR format (passa tudo o que esteja a branco para azul)

    cv.imshow('Imagem Original', img)
    cv.imshow('Canal Azul', B)
    cv.imshow('Mascara', mask)
    cv.imshow('Mascara Azul', azul_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
