import torch
import numpy as np


def power_law_diag_matrix(P, exponent, device):

    k = torch.arange(1, P+1, device=device, dtype=torch.float64) / P
    diag_entries = k**(-exponent)

    return torch.diag(diag_entries)

# This is the old version of the function used in the main text.
# def power_law_diag_matrix(size, exponent, device):
#     diag_entries = torch.tensor([i**exponent for i in range(1, size+1)], device=device, dtype=torch.float64)
#     diag_entries = diag_entries / torch.sum(diag_entries)
#     diag_entries = diag_entries * size
#     return torch.diag(diag_entries)


def sample_power_law_matrix(P, N, exponent, device):

    cov = torch.sqrt(power_law_diag_matrix(P, exponent, device))

    R = torch.randn(P, N, device=device, dtype=torch.float64) / np.sqrt(N)

    act = cov @ R

    return act @ act.T, np.diag(cov.cpu().numpy())**2


def i_over_P(eig, exponent, N, P):

    q = P / N
    eig = np.abs(eig)

    gamma = 1/exponent
    x = eig**(-gamma)

    cot = (np.cos(np.pi*gamma)/np.sin(np.pi*gamma))

    correction = x*np.pi*cot - x**(1/gamma)/(gamma - 1) - x**(2/gamma)/(gamma - 2) - x**(
        3/gamma)/(gamma - 3) - x**(4/gamma)/(gamma - 4) - x**(5/gamma)/(gamma - 5)

    lhs = x*(1 - q*gamma**2*correction)

    return lhs


def eqn(eig, i, P, N, exponent):

    c = i / P
    lhs = i_over_P(eig, exponent, N,  P)

    return (lhs - c)**2


def power_law_theory(P, N, exponent):

    from scipy.optimize import minimize

    eigs = []
    for i in range(1, P+1):
        x0 = (i / P)**(-exponent)
        eig = minimize(eqn, x0, args=(i, P, N, exponent),  tol=1e-15).x
        eigs += [eig]

    return np.array(eigs).squeeze()


def infer_power_law_exponent(eigs, N=None, cutoff=10):

    from scipy.optimize import curve_fit
    from functools import partial
    P = len(eigs)

    x = eigs[:cutoff]
    y = np.arange(1, P+1)[:cutoff]/P

    if N is not None:
        fn = partial(i_over_P, P=P, N=N)
        bounds = ((1, ), (np.inf,))
    else:
        fn = partial(i_over_P, P=P)
        bounds = ((1, 1), (np.inf, np.inf))

    popt, pcov = curve_fit(fn, x, y, bounds=bounds, nan_policy='omit')

    return popt
