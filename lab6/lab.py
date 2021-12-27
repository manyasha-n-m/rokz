import numpy as np
import cv2


class BinSum:
    def __init__(self, img, e0, e1):
        self.img = img
        self.e0 = e0
        self.e1 = e1
        self.h, self.w = e0.shape
        self.gh = self.create_gh()
        self.gv = self.create_gv()
        self.g = self.create_g()

    def create_gh(self):
        pass

    def create_gv(self):
        pass

    def create_g(self):
        pass
    