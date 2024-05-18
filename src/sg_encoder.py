import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
from scipy.sparse import csr_matrix, spdiags, find


def read_graph(path):
    '''
    Read the signed network from the path

    inputs
        path: string
            path for input file

    outputs
        A: csr_matrix
            signed adjacency matrix
        base: int
            base number of node id (start from base)
    '''

    X = np.loadtxt(path, dtype=float, comments='#')
    m, n = X.shape

    if n <= 2 or n >= 4:
        raise FormatError('Invalid input format')

    base = np.amin(X[:, 0:2]) # 冒号遵循左闭右开原则，base是为了确定节点编号从0开始或从1开始

    if base < 0:
        raise ValueError('Out of range of node id: negative base')

    X[:, 0:2] = X[:, 0:2] - base
    rows = X[:, 0]
    cols = X[:, 1]
    data = X[:, 2]

    n = int(np.amax(X[:, 0:2]) + 1)

    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    return A, int(base)


def semi_row_normalize(A):
    '''
    Perform the semi row-normalization for given adjacency matrix

    inputs
        A: csr_matrix
            adjacency matrix of given graph

    outputs
        nAp: csr_matrix
            positive semi row-normalized adjacency matrix
        nAn: csr_matrix
            negative semi row-normalized adjacency matrix
    '''

    m, n = A.shape

    # row-wise sum, d is out-degree for each node
    d = abs(A).sum(axis=1)
    d = np.asarray(d).flatten()

    d = np.maximum(d, np.ones(n))
    invd = 1.0 / d
    invD = spdiags(invd, 0, m, n)
    snA = invD * A

    I, J, K = find(snA)

    pos = K > 0
    neg = K < 0

    nAp = csr_matrix((abs(K[pos]), (I[pos], J[pos])), shape=(m, n))
    nAn = csr_matrix((abs(K[neg]), (I[neg], J[neg])), shape=(m, n))

    return nAp, nAn


def iterate(nApT, nAnT, seed, c, epsilon, beta, gamma, max_iters,
            handles_deadend, verbose):
    '''
    Perform power iteration for SRWR query

    inputs
        nApT: csr_matrix
            positive semi row-normalized adjacency matrix (transpose)
        nAnT: csr_matrix
            negative semi row-normalized adjacency matrix (transpose)
        seed: int
            seed (query) node
        c: float
            restart probability
        epsilon: float
            error tolerance for power iteration
        beta: float
            balance attenuation factor
        gamma: float
            balance attenuation factor
        max_iters: int
            maximum number of iterations for power iteration
        handles_deadend: bool
            if true, it will handle the deadend issue in power iteration
            otherwise, it won't, i.e., no guarantee for sum of SRWR scores
            to be 1 in directed graphs
        verbose: bool
            if true, it will show a progress bar over iterations

    outputs:
        rd: ndarray
            relative trustworthiness score vector w.r.t. seed
        rp: ndarray
            positive SRWR vector w.r.t. seed
        rn: ndarray
            negative SRWR vector w.r.t. seed
        residuals: list
            list of residuals of power iteration,
            e.g., residuals[i] is i-th residual
    '''

    m, n = nApT.shape
    q = np.zeros((n, 1))
    q[seed] = 1.0 # 输入的种子数不能超过数据集中的节点数

    rp = q
    rn = np.zeros((n, 1))
    rt = np.row_stack((rp, rn))

    residuals = np.zeros((max_iters, 1))

    # pbar = tqdm(total=max_iters, leave=True, disable=not verbose)
    for i in range(max_iters):
        if handles_deadend:
            new_rp = (1 - c) * (nApT.dot(rp + (1.0 - gamma) * rn) +
                                beta * (nAnT.dot(rn)))
            new_rn = (1 - c) * (gamma * (nApT.dot(rn)) +
                                nAnT.dot(rp + (1.0 - beta) * rn))
            P = np.sum(new_rp) + np.sum(new_rn)
            new_rp = new_rp + (1.0 - P) * q
        else:
            new_rp = (1 - c) * (nApT.dot(rp + (1.0 - gamma) * rn) +
                                beta * (nAnT.dot(rn))) + c * q
            new_rn = (1 - c) * (gamma * (nApT.dot(rn)) +
                                nAnT.dot(rp + (1.0 - beta) * rn))

        new_rt = np.row_stack((new_rp, new_rn))

        residuals[i] = norm(new_rt - rt, 1)

        # pbar.set_description("Residual at %d-iter: %e" % (i, residuals[i]))
        if residuals[i] <= epsilon:
            # pbar.set_description("SRWR scores have converged")
            # pbar.update(max_iters)
            break

        rp = new_rp
        rn = new_rn
        rt = new_rt

    rd = rp - rn

    return rd, rp, rn, residuals


class SRWR:
    normalized = False

    def __init__(self):
        pass

    def read_graph(self, input_path):
        '''
        Read a graph from the given input path
        This function performs the normalization as well

        inputs
            input_path: string
                path for input file
        '''

        self.A, self.base = read_graph(input_path)
        self.d = abs(self.A).sum(axis=1) # 计算每个节点的出度
        self.normalize()

    def normalize(self):
        '''
        Normalize the given graph
        '''

        if self.normalized is False:
            # self.nAp, self.nAn = normalizer.semi_row_normalize(self.A)
            self.nAp, self.nAn = semi_row_normalize(self.A)
            self.nApT = self.nAp.T
            self.nAnT = self.nAn.T
            self.normalized = True

    def query(self, seed, c=0.15, epsilon=1e-9, beta=0.5, gamma=0.5,
              max_iters=300, handles_deadend=True, verbose=True):
        '''
        Compute an SRWR query for given seed

        inputs
            seed: int
                seed (query) node
            c: float
                restart probability
            epsilon: float
                error tolerance for power iteration
            beta: float
                balance attenuation factor
            gamma: float
                balance attenuation factor
            max_iters: int
                maximum number of iterations for power iteration
            handles_deadend: bool
                if true, it will handle the deadend issue in power iteration
                otherwise, it won't, i.e., no guarantee for sum of SRWR scores
                to be 1 in directed graphs
            verbose: bool
                if true, it will show a progress bar over iterations

        outputs:
            rd: ndarray
                relative trustworthiness score vector w.r.t. seed
            rp: ndarray
                positive SRWR vector w.r.t. seed
            rn: ndarray
                negative SRWR vector w.r.t. seed
            residuals: list
                list of residuals of power iteration,
                e.g., residuals[i] is i-th residual
        '''

        rd, rp, rn, residuals = iterate(self.nApT, self.nAnT, seed, c,
                                                 epsilon, beta, gamma,
                                                 max_iters, handles_deadend,
                                                 verbose)

        return rd, rp, rn, residuals


def write_vectors(rd, rp, rn, mx, seed, output_type):
    '''
    Write vectors into a file
    '''
    if output_type is 'rp':
        X = rp
    elif output_type is 'rn':
        X = rn
    elif output_type is 'rd':
        X = rd
    elif output_type is 'both':
        X = np.column_stack((rp, rn))
    else:
        raise ValueError('Type of output should be {rp, rn, rd, both}')

    X = X.flatten() #把列向量转为行向量
    mx[seed] = X


def process_query(input_path, output_path, output_type, c=0.15, epsilon=1e-9,
                  beta=0.5, gamma=0.5, max_iters=300, handles_deadend=True):
    '''
    Processed a query to obtain a score vector w.r.t. the seeds

    inputs
        input_path : str
            path for the graph data
        output_path : str
            path for storing an RWR score vector
        output_type : str
            type of output {'rp', 'rn', 'rd', 'both'}
                * rp: a positive SRWR score vector
                * rn: a negative SRWR score vector
                * rd: a trusthworthiness vector, i.e., rd = rp - rn
                * both: both of rp and rn (1st column: rp, 2nd column: rn)
        seed : int
            seed for query
        c : float
            restart probability
        epsilon : float
            error tolerance for power iteration
        beta : float
            balance attenuation factor
        gamma : float
            balance attenuation factor
        max_iters : int
            maximum number of iterations for power iteration
        handles_deadend : bool
            if true, it will handle the deadend issue in power iteration
            otherwise, it won't, i.e., no guarantee for sum of RWR scores
            to be 1 in directed graphs
    '''

    srwr = SRWR()
    srwr.read_graph(input_path)
    srwr.normalize()
    num = np.size(srwr.A, 0)
    mx = np.zeros((num, num))
    for seed in range(num):
        rd, rp, rn, residuals = srwr.query(seed, c, epsilon, beta, gamma, max_iters, handles_deadend)
        write_vectors(rd, rp, rn, mx, seed, output_type)
    np.savetxt(output_path, mx)
    return mx