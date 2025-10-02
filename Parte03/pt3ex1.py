from copy import deepcopy
import numpy as np
import cv2 as cv
import argparse # For command line arguments


def main():
    pass

    # Command line arguments, to easily change input video file
    parser = argparse.ArgumentParser(
        prog='Traffic car couter',
        description='Counts cars',
        epilog='End of the program'
    )

    parser.add_argument('-if', '--input_file', type=str, default='docs/traffic.mp4')

    args = vars(parser.parse_args())
    print(args)

    # Open the video file
    cap = cv.VideoCapture(args['input_file'])

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        print("Video opened successfully.")
    

    lista_media_1 = []
    lista_media_2 = []
    lista_media_3 = []
    lista_media_4 = []

    j = 0

    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    counter_4 = 0

    # Read and display all frames
    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        # Convert to grayscale
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


        # --- Via 1 ---

        # Read the values of a vector of pixels x = 90:310
        pixel_values_1 = []
        for i in range(90, 310):
            pixel_value_1 = frame_gray[718,i]
            pixel_values_1.append(pixel_value_1)
        media_1 = round(np.mean(pixel_values_1),2)
        #print(media_1)

        lista_media_1.append(media_1)
        #print(lista_media_1)

        if j > 1 and lista_media_1[j-1] - lista_media_1[j-2] < 30:
            diff = lista_media_1[j] - lista_media_1[j-1]
            if diff < -30:
                counter_1 += 1
                print("Carros Detetados na Via 1:", counter_1)
        

        # --- Via 2 ---

        # Read the values of a vector of pixels x = 370:600
        pixel_values_2 = []
        for i in range(370, 600):
            pixel_value_2 = frame_gray[718,i]
            pixel_values_2.append(pixel_value_2)
        media_2 = round(np.mean(pixel_values_2),2)
        #print(media_2)

        lista_media_2.append(media_2)
        #print(lista_media_2)

        if j > 1 and lista_media_2[j-1] - lista_media_2[j-2] < 30:
            diff = lista_media_2[j] - lista_media_2[j-1]
            if diff < -30:
                counter_2 += 1
                print("Carros Detetados na Via 2:", counter_2)


        # --- Via 3 ---

        # Read the values of a vector of pixels x = 674:950
        pixel_values_3 = []
        for i in range(674, 950):
            pixel_value_3 = frame_gray[718,i]
            pixel_values_3.append(pixel_value_3)
        media_3 = round(np.mean(pixel_values_3),2)
        #print(media_3)

        lista_media_3.append(media_3)
        #print(lista_media_3)

        if j > 1 and lista_media_3[j-1] - lista_media_3[j-2] < 30:
            diff = lista_media_3[j] - lista_media_3[j-1]
            if diff < -30:
                counter_3 += 1
                print("Carros Detetados na Via 3:", counter_3)

        
        # --- Via 4 ---

        # Read the values of a vector of pixels x = 990:1240
        pixel_values_4 = []
        for i in range(990, 1240):
            pixel_value_4 = frame_gray[718,i]
            pixel_values_4.append(pixel_value_4)
        media_4 = round(np.mean(pixel_values_4),2)
        #print(media_4)

        lista_media_4.append(media_4)
        #print(lista_media_4)

        if j > 1 and lista_media_4[j-1] - lista_media_4[j-2] < 30:
            diff = lista_media_4[j] - lista_media_4[j-1]
            if diff < -30:
                counter_4 += 1
                print("Carros Detetados na Via 4:", counter_4)


        # Apresentar contador na imagem
        cv.putText(frame, f'Carros na Via 1: {counter_1}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f'Carros na Via 2: {counter_2}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f'Carros na Via 3: {counter_3}', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f'Carros na Via 4: {counter_4}', (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv.imshow('Frame', frame)
        key = cv.waitKey(50)
        if key == 113: # ASCII for 'q' to quit
            break


        j += 1

    cv.destroyAllWindows()







if __name__ == "__main__":
    main()