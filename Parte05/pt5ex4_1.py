import cv2 as cv
import numpy as np
import argparse
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('QtAgg') 
import json # Use QtAgg backend for better interactivity


def main():

 
    parser = argparse.ArgumentParser(prog='Optimization for a line')
    parser.add_argument('-f', '--filename', type=str, default='./data_4.json')
    parser.add_argument('-n', '--num_points', type=int, default=30, help='Number of points to collect')
    args = vars(parser.parse_args())
    print(args)
                                     
    plt.axis([-10, 10,-5, 5])

    points = plt.ginput(n=args['num_points'])
    print("Pontos recolhidos:", points)

    # Lists to store the x and y coordinates of the points
    xs = []
    ys = [] 

    for point in points:
        x = float(point[0])
        y = float(point[1])
        xs.append(x)
        ys.append(y)

    # Save the collected points in a dictionary
    data = {'xs': xs, 'ys': ys}
    print("Pontos recolhidos:", data)



    with open(args['filename'], "w") as f:
        f.write(json.dumps(data, indent=0))




if __name__ == "__main__":
    main()