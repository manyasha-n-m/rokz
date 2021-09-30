import numpy as np

class EMalg:
    """
    >>> import numpy as np
    >>> from keras.datasets import mnist
    >>> (train_X, train_y), (test_X, test_y) = mnist.load_data()
    ...
    >>> X = train_X[(train_y==0)|(train_y==1)]
    >>> y = train_y[(train_y==0)|(train_y==1)]
    >>> X_new = np.where(X<50, 0, X) 
    >>> X_new = np.where(X_new>0, 1, X_new) 
    >>> E = EMalg(np.asarray(X_new[:4]), 2)
    >>> E.pkx = np.array([[0.25, 0.75], [0.73, 0.27], [0.4, 0.6], [0.33, 0.67]])
    >>> E.pk_new(0)
    0.4275
    >>> E.maximize()
    >>> E.calc_pkx(E.X[0],1)
    1.0
    """
    def __init__(self, X, k_len):
        self.X = X
        self.K = np.arange(k_len)
        self.pkx = self.pkx_init()
        
    def pkx_init(self):
        # sum_k(pkx(k|xi)) = 1
        return np.array([np.random.dirichlet(np.ones(len(self.K)),size=1)[0] for i in range(len(self.X))])
    
    def pk_new(self, k):
        return np.sum(self.pkx[:,k])/len(self.X)
    
    def calc_pij(self, k):
        pkx = self.pkx[:,k]
        pij = np.array([[np.sum(pkx*self.X[:,i,j])/np.sum(pkx) for j in range(28)] for i in range(28)])
        return pij
    
    def calc_pkx(self,X, k):
        Pij = np.where(X!=1, 1, self.pij[k])
        Inv_Pij = np.where(X!=0,1,1-self.pij[k])
        s = 0
        for k_ in self.K:
            if k_ == k:
                continue
            A = np.where(X!=1, 1, self.pij[k_])*np.where(X!=0,1,1-self.pij[k])/(Pij*Inv_Pij)
            s+=np.prod(A)*self.pk[k_]/self.pk[k]
        pkx = 1/(1 + s)
        return pkx
    
    def maximize(self):
        pij, pk = [], []
        for k in self.K:
            pk.append(self.pk_new(k))
            pij.append(self.calc_pij(k)) 
    
        self.pk = pk
        self.pij = pij
    
    def alg(self):
        for i in range(3):
            self.maximize()
            self.pkx=np.array([[self.calc_pkx(x,k) for k in self.K] for x in self.X])