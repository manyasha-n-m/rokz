import numpy as np
import cv2
from glob import glob

img_ = glob("**/sample_0.png", recursive=True)[0]
e0_ = glob("**/e0.png", recursive=True)[0]
e1_ = glob("**/e1.png", recursive=True)[0]


class BinSum:
    """
    >>> s = BinSum(img_, e0_, e1_)
    >>> s.gh('i','v0', 'i')
    True
    >>> s.gv('d0','d1', 'A00')
    False
    >>> s.g(1, 'r1')
    True
    >>> (s.block(0, 0, 0, 0)==s.e1).all()
    True
    """
    def __init__(self, img_path, e0_path, e1_path):
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.e0 = cv2.imread(e0_path, cv2.IMREAD_GRAYSCALE)
        self.e1 = cv2.imread(e1_path, cv2.IMREAD_GRAYSCALE)
        self.h, self.w = self.e0.shape
        self.cols = int(len(self.img[0, :])/self.w)
        self.T =[0, 1]
        self.N = self.__create_n()
        self.__q = {0: self.e0, 1: self.e1}
        self.__gh = self.__create_gh()
        self.__gv = self.__create_gv()
        self.__g = self.__create_g()
        self.f = self.calc_f0()

    def img_show(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.img)
        return plt.show()

    def q(self, t, x):
        if x.shape == (self.h, self.w):
            return (x == self.__q[t]).all()
        return False

    @staticmethod
    def __create_n():
        N = ['i', 'i_']
        for i in range(2):
            N.extend([f'v{i}', f'v{i}_', f'r{i}', f'd{i}'])
            for j in range(2):
                N.append(f'A{i}{j}')
        return N

    @staticmethod
    def __create_gh():
        gh = dict()
        gh['i', 'v0', 'i'] = gh['i_', 'v0_', 'i'] = True
        gh['i', 'v1', 'i_'] = gh['i_', 'v1_', 'i_'] = True
        gh['v0', 'v0', 'i'] = gh['v0_', 'v0', 'i'] =gh['v1', 'v0_', 'i'] = gh['v1_', 'v0_', 'i'] = True
        gh['v0', 'v1', 'i_'] = gh['v0_', 'v1', 'i_'] =gh['v1', 'v1_', 'i_'] = gh['v1_', 'v1_', 'i_'] = True
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

    def block(self, i, i_, j, j_) -> np.ndarray:
        st_i = i * self.h
        end_i = (i_ + 1)*self.h
        st_j = j * self.w
        end_j = (j_ + 1)*self.w
        return self.img[st_i:end_i, st_j: end_j]

    def calc_f0(self):
        f = dict()
        for i in range(3):
            for j in range(self.cols):
                for i_ in range(i, 3):
                    for j_ in range(j, self.cols):
                        for t in self.T:
                            f[i, i_, j, j_, t] = self.q(t, self.block(i, i_, j, j_))
                        for n in self.N:
                            f[i, i_, j, j_, n] = False
        return f

    def horizontal(self, i, i_, j, j_, n):
        if j_ == j:
            return False
        res = []
        for j__ in range(j, j_):
            for nl in self.N:
                for nr in self.N:
                    res.append(all([
                        self.f[i, i_, j, j__, nl],
                        self.gh(nl, nr, n),
                        self.f[i, i_, j__+1, j_, nr]]))
        return any(res)

    def vertical(self, i, i_, j, j_, n):
        if i_ == i:
            return False
        res = []
        for i__ in range(i, i_):
            for nu in self.N:
                for nd in self.N:
                    res.append(all([
                        self.f[i, i__, j, j_, nu],
                        self.gv(nu, nd, n),
                        self.f[i__+1, i_, j, j_, nd]]))
        return any(res)

    def rename(self, i, i_, j, j_, n):
        res = []
        for t in self.T:
            res.append(self.f[i, i_, j, j_, t] and self.g(t, n))
        return any(res)

    def iter_S1(self):
        for i in range(3):
            for j in range(self.cols):
                for n in self.N:
                    self.f[i, i, j, j, n] = any([self.f[i, i, j, j, n], self.rename(i, i, j, j, n)])

    def iter_S2(self):
        i, i_ = 0, 1
        for j in range(self.cols):
            for n in self.N:
                self.f[i, i_, j, j, n] = any([
                    self.f[i, i_, j, j, n],
                    self.vertical(i, i_, j, j, n),
                    self.rename(i, i_, j, j, n)
                ])

    def iterate(self, idx):
        i, i_ = 0, 2
        for j in range(self.cols + 1 - idx):
            j_ = idx +j - 1
            for n in self.N:
                self.f[i, i_, j, j_, n] = any([
                    self.f[i, i_, j, j_, n],
                    self.vertical(i, i_, j, j_, n),
                    self.horizontal(i, i_, j, j_, n),
                    self.rename(i, i_, j, j_, n)
                ])

    def check(self):
        self.iter_S1()
        self.iter_S2()
        for idx in range(1, self.cols+1):
            self.iterate(idx)
        return self.f

    def belongs(self):
        f = self.check()
        if f[0, 2, 0, self.cols-1, 'i']:
            for n_ in self.N:
                if f[0, 2, 0, 0, n_]:
                    return print('Belongs:', n_)
        return print('Doesn`t belong!')

