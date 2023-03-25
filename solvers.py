import numpy as np
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from utils import *
import networkx as nx

class GraphLassoParam(GraphicalLasso):
    def __init__(self, alpha=0.01, *, mode="cd", tol=0.0001, enet_tol=0.0001, max_iter=100, verbose=False, assume_centered=True, partcorr_precision=20):
        super().__init__(alpha, mode=mode, tol=tol, enet_tol=enet_tol, max_iter=max_iter, verbose=verbose, assume_centered=assume_centered)
        
        self.partcorr_precision = partcorr_precision
        self.name = 'Graphical Lasso Parametrized'
        
    def fit(self, X, y=None):
        res = super().fit(X, y)
        
        self.partcorr_ = np.around(pcorr(self.precision_,
                                         prec_passed=True),
                                   self.partcorr_precision)

        adjacency = (self.partcorr_ != 0).astype(np.int64)
        self.cgraph = nx.from_numpy_array(adjacency)

        return res
        
class GraphLasso(GraphicalLassoCV):
    def __init__(self, *, alphas=4, n_refinements=4, cv=None, tol=0.0001,
                enet_tol=0.0001, max_iter=100, mode="cd", n_jobs=-1, verbose=False, assume_centered=True, partcorr_precision=20):
        super().__init__(alphas=alphas,
                         n_refinements=n_refinements,
                         cv=cv,tol=tol,
                         enet_tol=enet_tol,
                         max_iter=max_iter,
                         mode=mode,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         assume_centered=assume_centered)
        self.partcorr_precision = partcorr_precision
        self.name = 'Graphical Lasso'
        
    def fit(self, X, y=None):
        res = super().fit(X, y)
        
        self.partcorr_ = np.around(pcorr(self.precision_,
                                         prec_passed=True),
                                   self.partcorr_precision)

        adjacency = (self.partcorr_ != 0).astype(np.int64)
        self.cgraph = nx.from_numpy_array(adjacency)

        return res
    
    
class MHT:
    def __init__(self, alpha=0.05, adjustment=None):
        self.methods = [None, 'bonferroni', 'sidak', 'holm', 'hochberg', 'benjamini-hochberg']
        self.alpha = alpha
        self.adjustment = adjustment if adjustment in self.methods else None
        self.name = self.adjustment if self.adjustment != None else 'None'
    
    def fit(self, X, y=None):
        self.n, self.N = X.shape
        self.part = pcorr(X)
        num_tests = self.N * (self.N - 1) / 2
        
        pvalues = test_graph(self.part, self.n)
        
        if self.adjustment == None:
            tested = pvalues <= self.alpha
            
        elif self.adjustment == 'bonferroni':
            significance = self.alpha / num_tests
            tested = pvalues <= significance
        
        elif self.adjustment == 'sidak':
            significance = 1 - ((1 - self.alpha) ** (1 / num_tests))
            tested = pvalues <= significance
        
        elif self.adjustment == 'holm':
            p_sorted = np.sort(pvalues[np.triu_indices(self.N, 1)], axis=None)
            for k in range(p_sorted.shape[0]):
                if p_sorted[k] > (self.alpha / (num_tests + 1 - (k + 1))):
                    break
            tested = np.zeros_like(self.part)    
            for i in range(k):
                idx = np.array(np.where(pvalues == p_sorted[i])).T
                index = None
                for x in idx:
                    if tested[tuple(x)] == 0:
                        index = tuple(x)
                        break
                tested[index] = 1
                tested[index[::-1]] = 1
                
        elif self.adjustment == 'hochberg':
            p_sorted = np.sort(pvalues[np.triu_indices(self.N, 1)], axis=None)
            R = 0
            for k in range(p_sorted.shape[0]):
                if p_sorted[k] <= (self.alpha / (num_tests + 1 - (k + 1))):
                    if k > R: R = k
            tested = np.zeros_like(self.part)
            for i in range(R + 1):
                idx = np.array(np.where(pvalues == p_sorted[i])).T
                index = None
                for x in idx:
                    if tested[tuple(x)] == 0:
                        index = tuple(x)
                        break
                tested[index] = 1
                tested[index[::-1]] = 1      
        
        elif self.adjustment == 'benjamini-hochberg':
            p_sorted = np.sort(pvalues[np.triu_indices(self.N, 1)], axis=None)
            R = 0
            for k in range(p_sorted.shape[0]):
                if p_sorted[k] <= (self.alpha * (k + 1) / num_tests):
                    if k > R: R = k
            tested = np.zeros_like(self.part)  
            for i in range(R + 1):
                idx = np.array(np.where(pvalues == p_sorted[i])).T
                index = None
                for x in idx:
                    if tested[tuple(x)] == 0:
                        index = tuple(x)
                        break
                tested[index] = 1
                tested[index[::-1]] = 1
        
        self.cgraph = nx.from_numpy_array(tested)
        
        return self
        