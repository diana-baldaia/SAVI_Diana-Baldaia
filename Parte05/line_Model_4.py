import cv2 as cv
import numpy as np
import argparse
import random
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('QtAgg') 
import json # Use QtAgg backend for better interactivity



class LineModel:

    def __init__(self, m=None, b=None):

        self.m = m
        self.b = b
        self.plot_handle = None

        if self.m is None or self.b is None:
            self.randomizeParameters()
    

    def randomizeParameters(self):
        self.m = random.uniform(-2, 2)
        self.b = random.uniform(0, 10)


    def draw(self, color='b'):

        xs = [-10,10]
        ys = self.getYs(xs)
        
        if self.plot_handle is None:
            self.plot_handle = plt.plot(xs, ys, '-'+color, linewidth=2)
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


    def __str__(self):
        return "Line m= " + str(self.m) + " b= " + str(self.b)