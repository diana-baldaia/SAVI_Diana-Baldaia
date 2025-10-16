import cv2 as cv
import numpy as np
import argparse
import random
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('QtAgg') 
import json # Use QtAgg backend for better interactivity
from line_Model_4 import LineModel
from parabola_Model import ParabolaModel
from segmented import SegmentedModel
from scipy.optimize import least_squares
 

def main():


    parser = argparse.ArgumentParser(prog='Optimization for a line')
    parser.add_argument('-f', '--filename', type=str, default='./data_4.json')
    parser.add_argument('-ni', '--number_iterations', type=str, default=500)
    args = vars(parser.parse_args())
    print(args)
                                     
    plt.axis([-10, 10,-10, 10])
    plt.grid()


    with open(args['filename'], "r") as file:
        data = json.load(file)

    xs_gt = data['xs']
    ys_gt = data['ys']
    plt.plot(xs_gt, ys_gt, '.k', markersize=12)

    segmented = SegmentedModel()
    

    def objectiveFunction(params):

        segmented.line_model.m = params[0]
        segmented.line_model.b = params[1]
        segmented.parabola_model.a = params[2]
        segmented.parabola_model.h = params[3]
        segmented.parabola_model.k = params[4]

        error = segmented.getError(xs_gt=xs_gt, ys_gt=ys_gt)
        print('error = ' + str(error))

        segmented.draw()
        plt.draw()
        plt.waitforbuttonpress(0.1)

        return error


    plt.draw()

    result = least_squares(objectiveFunction, [segmented.line_model.m, segmented.line_model.b, segmented.parabola_model.a, segmented.parabola_model.h, segmented.parabola_model.k])
    print('Raw result of least squares = ' + str(result))

    print('Solution: m= ' + str(result.x[0]) + ' b= ' + str(result.x[1]))
    plt.show()



if __name__ == "__main__":
    main()