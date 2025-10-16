import cv2 as cv
import numpy as np
import argparse
import random
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('QtAgg') 
import json # Use QtAgg backend for better interactivity


# y = a(x-h) ^ 2 + k
# a é a abertura
# h é o x do vértice
#k é o y do vértice

class ParabolaModel:

    def __init__(self, a=None, h=None, k=None):

        self.a = a
        self.h = h
        self.k = k
        self.plot_handle = None

        if self.a is None or self.h is None or self.k is None:
            self.randomizeParameters()

    
    def randomizeParameters(self):
        
        self.a = random.uniform(-1, 1)
        self.h = random.uniform(-10, 0)
        self.k = random.uniform(-5, 5)


    def draw(self, color='r'):

        xs = [-10, 0]
        ys = self.getYs(xs)

        if self.plot_handle is None:
            self.plot_handle = plt.plot(xs, ys,'-'+color, linewidth=2)
        else:
            plt.setp(self.plot_handle, xdata=xs, ydata=ys)
    

    def getYs(self, xs):

        ys = []
        for x in xs:
            y = self.a * (x - self.h)**2 + self.k
            ys.append(y)
        return ys
    

    def getError(self, xs_gt, ys_gt):

        ys = self.getYs(xs_gt)
        errors = [abs(ys_gt - y) for y, y_get in zip(ys, ys_gt)]
        return np.mean(errors)
    

    def __str__(self):
        return f"Parabola a={self.a:.2f} h={self.h:.2f} k={self.k:.2f}"