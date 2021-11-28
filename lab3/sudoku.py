import numpy as np
import copy

class sudoku:
    '''
    >>> arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> sudoku.complement(1,2, arr)
    array([1, 2, 7, 8])
    
    >>> tau, g = sudoku.create_tau(2)
    >>> tau['00']
    array(['01', '02', '03', '10', '20', '30', '11'], dtype='<U2')
    
    >>> g['00']['01'][0].astype(int)
    array([0, 1, 1, 1])
    
    >>> d3 = [[1, 0, 0, 3], [0, 3, 0, 0], [4, 0, 0, 0], [3, 0, 0, 0]]
    >>> q = sudoku.create_q(2, d3)
    >>> q['11'].astype(int)
    array([0, 0, 1, 0])
    
    >>> old = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    >>> qt = np.array([0, 0, 1, 0])
    >>> qt_ = np.array([1, 0, 0, 0])
    >>> sudoku.new_gt(qt, qt_, old).flatten()
    array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    '''
    def __init__(self, dim: int, base):
        self.dim = dim
        self.base = np.array(base)
        self.tau, self.g = sudoku.create_tau(dim) 
        self.q = sudoku.create_q(dim, self.base)
        
    @staticmethod
    def complement(i,j, small_block):
        A = np.delete(small_block, i, 0)
        return np.delete(A, j, 1).flatten()
    
    @classmethod
    def create_tau(cls, dim):
        n = dim*dim
        block = np.array([[f'{i}{j}' for j in range(n)] for i in range(n)])
        tau = {}
        g = {}
        for i in range(n):
            for j in range(n):
                # small_block_pos
                i_ = int(i/dim) 
                j_ = int(j/dim)
                small_block = block[i_*dim:i_*dim+dim, j_*dim:j_*dim+dim]
                tau[f'{i}{j}'] = np.concatenate([np.delete(block[i,:], j),
                                                np.delete(block[:,j], i),
                                                cls.complement(i%dim,j%dim, small_block)])
                g[f'{i}{j}'] = {}
                for near in tau[f'{i}{j}']:
                    g[f'{i}{j}'][near] = np.ones((n,n), dtype=bool)
                    np.fill_diagonal(g[f'{i}{j}'][near], False)
        return tau, g
    
    @staticmethod
    def create_q(dim, base):
        n = dim**2
        q = {}
        for i, v in enumerate(base):
            for j, e in enumerate(v):
                if e:
                    q[f'{i}{j}'] = np.zeros(n, dtype=bool)
                    q[f'{i}{j}'][e-1] = True
                else:
                    q[f'{i}{j}'] = np.ones(n, dtype=bool)
        return q
    
    @staticmethod
    def cut_g(g, q):
        same = True
        for t, neighbours in g.items():
            for t_, table in neighbours.items():
                _new = sudoku.new_gt(q[t], q[t_], table)
                if (g[t][t_] != _new).any() and same:
                    same = False
                g[t][t_] = _new
        return same
    
    @staticmethod
    def new_gt(qt, qt_, old):
        Q_, Q = np.meshgrid(qt_, qt)
        return old*Q_*Q
    
    @staticmethod
    def cut_q(q, g, base):
        same = True
        n = len(base)
        max_amounts = []
        for t, states in q.items():
            vertices = np.ones(n, dtype=bool)
            for table in g[t].values():
                vertices*=table.any(axis=1)
            _new = states * vertices 
            if same and (q[t] != _new).any():
                same = False
            q[t] = _new 
            
            amount = np.sum(q[t].astype(int))
            if amount == 0:
                return same, np.array([]), False
            elif amount == 1:
                i,j = t
                if base[int(i), int(j)] == 0:
                    base[int(i), int(j)] = list(q[t]).index(True)+1
                    print(base, t, list(q[t]).index(True)+1)
            else: 
                max_amounts.append([t, amount])
        return same, np.array(max_amounts), True
    
    @staticmethod
    def solve_case(base, g, q):
        solvable = True
        while solvable:
            same_g = sudoku.cut_g(g, q)
            same_q, amounts, solvable = sudoku.cut_q(q, g, base)
            if not solvable:
                print('NOT SOLVABLE CASE')
                return solvable, amounts, False 
            
            if same_q and same_g:
                # we need to figure out if it's the end
                if len(amounts):
                    return solvable, amounts, False
                else:
                    return solvable, amounts, True
    
    def solve_sudoku(self):
        solvable, amounts, solved = self.solve_case(self.base, self.g, self.q)
        poly = True
        while solvable:
            if solved:
                return solved, self.base, poly
            else:
                # we need to fix state in a node
                t, k = amounts[np.argsort(amounts[:,1])][0]
                pos1, pos2 = t
                available = np.arange(self.dim**2)[np.where(self.q[t])]
                for state in available:
                    poly = False
                    print(f'Fixing state {state+1} in position {t}')
                    g0 = copy.deepcopy(self.g)
                    q0 = copy.deepcopy(self.q)
                    base0 = copy.deepcopy(self.base)
                    
                    q0[t] = np.zeros(self.dim**2, dtype=bool)
                    q0[t][state] = True
                    base0[int(pos1), int(pos2)] = state+1
                    print(base0, t, state+1)
                    solvable1, amounts1, solved1 = self.solve_case(base0, g0, q0)
                    if not solvable1:
                        continue
                    else:
                        print('solvable case')
                        poly = True
                        break
                solvable = solvable1
                self.base = base0
                self.g = g0
                self.q = q0
                amounts = amounts1
                solved = solved1
        return solved, self.base, poly
    
    def solve(self):
        solved, base, poly = self.solve_sudoku()
        if not solved:
            print("No solution found")
            if not poly:
                print('No semilattice polymorphism either')
        else:
            print('Found solution')
            return base
            