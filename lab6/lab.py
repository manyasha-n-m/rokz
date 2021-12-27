import numpy as np
import cv2
from glob import glob

img_ = cv2.imread(glob("**/sample_0.png", recursive=True)[0], cv2.IMREAD_GRAYSCALE)
e0_ = cv2.imread(glob("**/e0.png", recursive=True)[0], cv2.IMREAD_GRAYSCALE)
e1_ = cv2.imread(glob("**/e1.png", recursive=True)[0], cv2.IMREAD_GRAYSCALE)

class BinSum:
    """
    >>> s = BinSum(img_, e0_, e1_)
    >>> s.gh('i','v0', 'i')
    True
    >>> s.gh('i','v0', 'i_')
    False
    """
    def __init__(self, img, e0, e1):
        self.img = img
        self.e0 = e0
        self.e1 = e1
        self.h, self.w = e0.shape
        self.T =[0, 1]
        self.N = self.__create_n()
        self.q = {0: e0, 1: e1}
        self.__gh = self.__create_gh()
        self.__gv = self.__create_gv()
        self.__g = self.__create_g()

    @staticmethod
    def __create_n():
        N = ['i', 'i_', '']
        for i in range(2):
            N.extend([f'v{i}', f'v{i}_', f'r{i}', f'd{i}'])
            for j in range(2):
                N.append(f'A{i}{j}')
        return N

    @staticmethod
    def __create_gh():
        gh = dict()
        gh['i', 'v0', 'i'] = gh['i_', 'v0_', 'i'] = gh['', 'v0', 'i'] = gh['', 'v0_', 'i'] = True
        gh['i', 'v1', 'i_'] = gh['i_', 'v1_', 'i_'] = gh['', 'v1', 'i_'] = gh['', 'v1_', 'i_'] = True
        return gh

    def gh(self, nl, nr, n):
        return (nl, nr, n) in self.__gh.keys()

    @staticmethod
    def __create_gv():
        gv = dict()
        gv['A00', 'r0', 'v0'] = gv['A01', 'r1', 'v0'] = gv['A10', 'r1', 'v0'] = True
        gv['A11', 'r0', 'v0_'] = True
        gv['A00', 'r1', 'v1'] = True
        gv['A01', 'r0', 'v1_'] = gv['A10', 'r0', 'v1_'] = gv['A11', 'r1', 'v1_'] = True
        for i in range(2):
            for j in range(2):
                gv[f'd{i}', f'd{j}', f'A{i}{j}'] = True
        return gv

    def gv(self, nu, nd, n):
        return (nu, nd, n) in self.__gv.keys()

    @staticmethod
    def __create_g():
        g = dict()
        for i in range(2):
            g[i, f'r{i}'] = True
            g[i, f'd{i}'] = True
        return g

    def g(self, t, n):
        return (t, n) in self.__g.keys()


