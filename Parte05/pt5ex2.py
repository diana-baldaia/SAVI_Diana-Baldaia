import cv2 as cv
import matplotlib
import numpy as np
import argparse
from copy import deepcopy
import glob
import random
from random import randint
from matplotlib import pyplot as plt
matplotlib.use('QtAgg') 
import json # Use QtAgg backend for better interactivity


class LineModel:

    def __init__(self, m=None, b=None):

        self.m = None
        self.b = None
        self.plot_handle = None

        if self.m is None or self.b is None:
            self.randomizeParameters()
    

    def randomizeParameters(self):
        self.m = random.uniform(-2, 2)
        self.b = random.uniform(-10, 10)


    def draw(self, color='b'):

        xs = [-10,10]
        ys = self.getYs(xs)
        
        if self.plot_handle is None:
            self.plot_handle = plt.plot(xs, ys, '-' + color, linewidth=2)
        else:
            plt.setp(self.plot_handle, xdata=xs, ydata = ys)


    def getYs(self,xs):

        ys = []
        for x in xs:
            y = self.m * x + self.b
            ys.append(y)
        
        return ys
    

    def getError(self, xs_gt, ys_gt):

        ys = self.getYs(xs_gt)

        errors = []
        for y, y_gt in zip(ys, ys_gt):
            error = abs(y_gt - y)
            errors.append(error)
        
        n =len(xs_gt)
        total = 0
        for error in errors:
            total += error
        
        average_error = total / n
        return average_error


def main():


    parser = argparse.ArgumentParser(prog='Optimization for a line')
    parser.add_argument('-f', '--filename', type=str, default='./data.json')
    parser.add_argument('-ni', '--number_iterations', type=int, default=500)
    args = vars(parser.parse_args())
    print(args)
                                     
    plt.axis([-10, 10,-10, 10])
    plt.grid()


    with open(args['filename'], "r") as file:
        data = json.load(file)


    xs_gt = data['xs']
    ys_gt = data['ys']
    plt.plot(xs_gt, ys_gt, '.k', markersize=12)

    line_model = LineModel()
    
    plt.draw()

    minimum_error = None
    best_line = LineModel()


    for i in range(0, args['number_iterations']):
        error = line_model.getError(xs_gt=xs_gt, ys_gt=ys_gt)
        print('Line' + str(i) + 'error = ' + str(error))

        if minimum_error is None or error < minimum_error:
            best_line.m = line_model.m
            best_line.b = line_model.b
            minimum_error = error


        line_model.draw()
        best_line.draw(color='r')
        plt.draw()
        plt.waitforbuttonpress(0.1)


        line_model.randomizeParameters()

    plt.show()



if __name__ == "__main__":
    main()