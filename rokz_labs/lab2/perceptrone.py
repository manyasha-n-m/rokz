import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

d1 = json.load(open('train_01.json'))
d = []
r = {'inside':1, 'outside':-1}
for k in d1.keys():
    _df = pd.DataFrame(d1[k], columns=['A', 'B'])
    _df['class'] = r[k]
    d.append(_df)
df1 = pd.concat(d, axis=0, ignore_index=True)

def plot(df):
    """
    >>> plot(d[0])
    Only one class was detected
    """
    plt.figure(figsize=(7,7))
    font = {'weight' : 'bold', 'size'   : 15}
    plt.rc('font', **font)
    try:
        df1, df2 = [x for _, x in df.groupby(df['class'] == -1)]
        plt.scatter(df1.A, df1.B)
        plt.scatter(df2.A, df2.B)
    except Exception:
        print('Only one class was detected')
        plt.scatter(df.A, df.B)
    plt.xlabel('A')
    plt.ylabel('B')
    plt.legend(['inside', 'outside'])

class Classifier:
    """
    >>> C = Classifier(df1)
    >>> C.check()
    (False, 0)
    >>> C.fix(C.X.iloc[0,:], C.y[0])
    >>> np.asarray(C.w)[0]
    0.4275968563344624
    """
    def __init__(self, train):
        self.df = train
        self.X = self.df[['A', 'B']]
        self.X['A2'] = self.X.A*self.X.A
        self.X['B2'] = self.X.B*self.X.B
        self.X['AB'] = self.X.A*self.X.B
        self.X['C'] = 1
        self.y = train['class']
        self.w = np.zeros(6)
    
    def fix(self, X, y):
        self.w += y*X/np.linalg.norm(X)
        
    def check(self):
        res = self.y * np.sum(self.w * self.X, axis=1)
        bad = self.X[res <= 0]
        try:
            return False, bad.index[0]
        except Exception:
            return True, 0
    
    def classify(self):
        flag, idx = self.check()
        while not flag:
            self.fix(self.X.loc[idx,:], self.y[idx])
            flag, idx = self.check()
        print(self.w)
    
    def train(self):
        self.classify()
        self.plot(self.df)
        
    def test(self, df):
        test = df[['A', 'B']]
        test['A2'] = test.A*test.A
        test['B2'] = test.B*test.B
        test['AB'] = test.A*test.B
        test['C'] = 1
        test['class_detected'] = (np.sum(self.w * test[['A', 'B', 'A2','B2','AB','C']], axis=1) > 0).astype(int)
        test['class_detected'].replace(0,-1,inplace=True)
        return test
    
    def plot(self, df):
        plot(df)
        xmax, xmin = np.max(df.A), np.min(df.A)
        ymax, ymin = np.max(df.B), np.min(df.B)
        x = np.linspace(xmin-0.1,xmax+0.1,100)
        y = np.linspace(ymin-0.1,ymax+0.1,100)
        X, Y = np.meshgrid(x,y)
        ones = np.ones((100,100))
        Z = np.zeros((100,100))
        for i, m in enumerate([X,Y,X*X,Y*Y, X*Y, ones]):
            Z+=self.w[i]*m
        plt.contour(X, Y, Z, [0])
