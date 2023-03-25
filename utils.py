import numpy as np
from scipy.stats import t
from sklearn.datasets import make_sparse_spd_matrix
import networkx as nx
from collections.abc import Iterable
from tqdm import tqdm

def pcorr(data, prec_passed=False):
    """Returns partial correlation matrix"""
    if not prec_passed:
        cov =  np.cov(data.T)
        prec = np.linalg.inv(cov)
    else:
        prec = data
    D = np.diag(np.sqrt(1 / prec.diagonal()))

    pc = -(D @ prec @ D)
    pc[np.diag_indices_from(pc)] *= -1
    
    return pc

def partcorr_test(r, n, k):
    """Returns p-value for partial correlation t-test"""
    dof = n - k - 2
    t_stat = r * np.sqrt(dof / (1 - r ** 2))
    
    return 2 * t.sf(np.abs(t_stat), dof)

def test_graph(partcorr, n):
    """Returns p-value of every edge with partial correlation t-test"""
    result = np.zeros_like(partcorr)
    for i in range(partcorr.shape[0]):
        for j in range(i + 1, partcorr.shape[0]):
            result[i, j] = partcorr_test(partcorr[i, j], n, partcorr.shape[0] - 2)
            
    return (result + result.T) + np.eye(result.shape[0])


class GraphDataset:
    def __init__(self, dim, zero_proba, size, generator, distribution='normal'):
        self.precision, self.covariance, self.graph = generator(dim, zero_proba)
        self.pos = nx.spring_layout(self.graph)
        if distribution == 'normal':
            self.samples = np.random.multivariate_normal(np.zeros(dim), self.covariance, size=size)
        #elif distribution == 't':
            #self.samples = scipy.random.multivariate_t(np.zeros(dim), self.covariance, df=3, size=size)

    def score(self, pred_graph, scorer):
        pred = pred_graph.edges
        true = self.graph.edges
        full = nx.complete_graph(self.precision.shape[0]).edges
        
        TP = len(pred & true)
        TN = len((full - pred) & (full - true))
        FP = len(pred & (full - true))
        FN = len((full - pred) & true)
        
        scores = {}
        
        if not isinstance(scorer, Iterable):
            scorer = [scorer]
        
        for sc in scorer:
            scores[sc.__name__] = sc(TP, TN, FP, FN)
            
        return scores
    
    def draw_graph(self):
        nx.draw(self.graph, self.pos)
        nx.draw_networkx_labels(self.graph, self.pos)
        
        
def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def precision(TP, TN, FP, FN):
    return TP / (TP + FP) if TP + FP != 0 else np.nan

def recall(TP, TN, FP, FN):
    return TP / (TP + FN) if TP + FN != 0 else np.nan

def f1(TP, TN, FP, FN):
    try:
        val = 2 * (precision(TP, TN, FP, FN) * recall(TP, TN, FP, FN)) / (precision(TP, TN, FP, FN) + recall(TP, TN, FP, FN))
        return val
    except:
        return np.nan
    
def fdr(TP, TN, FP, FN):
    try:
        val = 1 - precision(TP, TN, FP, FN)
        return val
    except:
        return np.nan
    

def perform_experiments(dim, zero_proba, observations, replications, solvers, generator, scorers=[f1, fdr], distribution='normal'):
    results = []
    for _ in tqdm(range(replications)):
        experiment = {}
        ds = GraphDataset(dim, zero_proba, observations, generator, distribution)
        for solver in solvers:
            solver.fit(ds.samples)
            predicted = solver.cgraph
            
            experiment[solver.name if solver.name is not None else 'None'] = ds.score(predicted, scorers)
        results.append(experiment)
        
    return results

def generate_peng(dim, er_proba, verbose=False):
    g = nx.fast_gnp_random_graph(dim, er_proba)
    if verbose:
        print(f'Generated graph density: {nx.density(g)}')
    s = nx.to_numpy_array(g)
    for i in range(dim):
        s[i] = s[i] * np.random.uniform(low=0.5, high=1, size=dim) * np.random.choice([1, -1], size=dim)
        #for j in range(dim):
        #    if i != j and s[i, j] != 0:
        #        val = np.random.uniform()
        #        s[i, j] = val - 1 if val < 0.5 else val
    s -= np.diag(np.diag(s))
    for i in range(dim):
        scaling = np.sum(np.abs(s[i]))
        s[i] /= 1.5 * scaling
        #for j in range(dim):
        #    if i != j and s[i, j] != 0:
        #        s[i, j] /= 1.5 * scaling
    s += np.eye(dim)
    s = (s + s.T) / 2
    if verbose:
        density = np.count_nonzero(s[np.tril_indices_from(s, k=-1)]) / (dim * (dim - 1) / 2)
        print(f'Got density: {density}')
        if np.all(np.linalg.eigvals(s) > 0): print("Matrix is symmetric PD")
        
    si = np.linalg.inv(s)
    D = np.diag(np.sqrt(np.diag(si)))
    return s, D @ si @ D, g 

def generate_chol(dim, alpha):
    prec = make_sparse_spd_matrix(dim, alpha=alpha, norm_diag=True)
    adj = (prec != 0).astype(int) - np.eye(dim)
    g = nx.from_numpy_array(adj)
    return prec, np.linalg.inv(prec), g

def sym_mat_density(mat):
    triu_elem = mat[np.triu_indices_from(mat, k=1)]
    return 1 - (len(triu_elem[triu_elem == 0]) / len(triu_elem))