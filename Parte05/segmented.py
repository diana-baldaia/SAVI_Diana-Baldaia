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


class SegmentedModel: 
        def __init__(self):
            self.parabola_model = ParabolaModel()
            self.line_model = LineModel()
            self.plot_handles = []

        def randomizeParameters(self):
            self.parabola_model.randomizeParameters()
            self.line_model.randomizeParameters()

        def draw(self):
            if self.plot_handles:
                for handle in self.plot_handles:
                    try:
                        if isinstance(handle, list):
                            handle[0].remove()
                        else:
                            handle.remove()
                    except Exception as e:
                        print(f"Erro ao remover handle: {e}")
                self.plot_handles = []

            xs_parabola = np.linspace(-10, 0, 100)
            ys_parabola = self.parabola_model.getYs(xs_parabola)
            self.plot_handles.extend(plt.plot(xs_parabola, ys_parabola, '-g', linewidth=2))

            xs_line = np.linspace(0, 10, 100)
            ys_line = self.line_model.getYs(xs_line)
            self.plot_handles.extend(plt.plot(xs_line, ys_line, '-b', linewidth=2))

        def getError(self, xs_gt, ys_gt):
            errors = []
            for x_gt, y_gt in zip(xs_gt, ys_gt):
                if x_gt < 0:
                    y_pred = self.parabola_model.a * (x_gt - self.parabola_model.h)**2 + self.parabola_model.k
                else:
                    y_pred = self.line_model.m * x_gt + self.line_model.b
                errors.append(y_gt - y_pred)
            return np.array(errors)

        def __str__(self):
            return f"Segmented Model:\n  {self.parabola_model}\n  {self.line_model}"