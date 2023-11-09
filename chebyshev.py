import numpy as np
from scipy import sparse


def compute_cheby_coeff(f, m=30, N=None, a_arange=None):
    """
    Compute Chebyshev coefficients for a Filterbank.

    Parameters
    ----------
    f : Filter
        Filterbank with at least 1 filter
    m : int
        Maximum order of Chebyshev coeff to compute
        (default = 30)
    N : int
        Grid order used to compute quadrature
        (default = m + 1)
    i : int
        Index of the Filterbank element to compute
        (default = 0)

    Returns
    -------
    c : ndarray
        Matrix of Chebyshev coefficients

    """
    # G = f.G
    # i = kwargs.pop('i', 0)

    if not N:
        N = m + 1
    if not a_arange:
        a_arange = [0, 2]

    # a_arange = [0, G.lmax]

    a1 = (a_arange[1] - a_arange[0]) / 2.0
    a2 = (a_arange[1] + a_arange[0]) / 2.0
    c = np.zeros(m + 1)

    tmpN = np.arange(N)
    num = np.cos(np.pi * (tmpN + 0.5) / N)
    for o in range(m + 1):
        c[o] = 2. / N * np.dot(f(a1 * num + a2),
                               np.cos(np.pi * o * (tmpN + 0.5) / N))

    return c


def cheby_op(L, c, signal, a_arange):
    """
    Chebyshev polynomial of graph Laplacian applied to vector.

    Parameters
    ----------
    G : Graph
    L : Laplacian Matrix of the graph
    c : ndarray or list of ndarrays
        Chebyshev coefficients for a Filter or a Filterbank
    signal : ndarray
        Signal to filter

    Returns
    -------
    r : ndarray
        Result of the filtering

    """
    # Handle if we do not have a list of filters but only a simple filter in cheby_coeff.
    if not isinstance(c, np.ndarray):
        c = np.array(c)

    c = np.atleast_2d(c)
    Nscales, M = c.shape
    N = signal.shape[0]

    if M < 2:
        raise TypeError("The coefficients have an invalid shape")

    # thanks to that, we can also have 1d signal.
    try:
        Nv = np.shape(signal)[1]
        r = np.zeros((N * Nscales, Nv))
    except IndexError:
        r = np.zeros((N * Nscales))

    a1 = float(a_arange[1] - a_arange[0]) / 2.
    a2 = float(a_arange[1] + a_arange[0]) / 2.

    twf_old = signal
    twf_cur = (np.dot(L, signal) - a2 * signal) / a1

    tmpN = np.arange(N, dtype=int)
    for i in range(Nscales):
        r[tmpN + N*i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur

    factor = 2/a1 * (L - a2 * sparse.eye(N))
    for k in range(2, M):
        twf_new = factor.dot(twf_cur) - twf_old
        for i in range(Nscales):
            r[tmpN + N*i] += c[i, k] * twf_new

        twf_old = twf_cur
        twf_cur = twf_new

    return r