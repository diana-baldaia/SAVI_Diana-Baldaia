from copy import deepcopy
import glob
import cv2 
import numpy as np
import argparse

def main():

    # ------------------------------------
    # Setup argparse
    # ------------------------------------
    parser = argparse.ArgumentParser(
        prog='Traffic car couter',
        description='Counts cars',
        epilog='This is finished')

    parser.add_argument('-if', '--image_filename', type=str, default='images/santorini/1.png')

    args = vars(parser.parse_args())
    print(args)

    # Load the image
    image = cv2.imread(args['image_filename'])
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(gray_image.shape)

    sift = cv2.SIFT_create(nfeatures=500)  # create an object for the sift detecting 500 keypoints
    key_points = sift.detect(gray_image, None) # detect the keypoints in the image

    print('key_points=\n' + str(key_points))

    gui_image = cv2.drawKeypoints(gray_image, key_points, image) # draw the keypoints in the image

    win_name = 'SIFT Keypoints'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, gui_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()