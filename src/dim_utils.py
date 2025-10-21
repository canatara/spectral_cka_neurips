import numpy as np
import torch
from torch.func import vmap

if not torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float64)
torch._dynamo.config.recompile_limit = 16  # Needed to avoid recompilation issues


def _get_device_dtype():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float64
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float64
    print(f"Using {device} with {dtype} precision")
    return device, dtype


DEVICE, DTYPE = _get_device_dtype()


def gett_all(ijlm: str, A, B):
    i, j, l, m = list(ijlm)
    pexp = i+'a,'+j+'a,'+l+'b,'+m+'b->'
    qexp = i+'a,'+j+'a,'+l+'a,'+m+'a->'
    pval = torch.einsum(pexp, A, B, A, B)
    pqval = pval - torch.einsum(qexp, A, B, A, B)
    return pval, pqval


@torch.compile(mode="max-autotune", fullgraph=False, dynamic=True)
def estimate_dimensionality(A: torch.Tensor, B: torch.Tensor):
    """
    Estimate the dimensionality of the population A.
    B is either the same as A or a different trial of the same population.
    Returns a (4,2) tensor:
        [[naive_num,   naive_den],
         [row_num,     row_den],
         [col_num,     col_den],
         [double_num,  double_den]]
    """
    P, Q = A.shape
    nf = (P * Q) ** 0.5

    t1, t1d = gett_all('ijji', A/nf, B/nf)
    t2, t2d = gett_all('iiii', A/nf, B/nf)
    t3, t3d = gett_all('ijjj', A/nf, B/nf)
    # t4, t4d = gett_all('iiij', An)#<
    t5, t5d = gett_all('ijjl', A/nf, B/nf)
    t6, t6d = gett_all('iijj', A/nf, B/nf)
    t7, t7d = gett_all('iijl', A/nf, B/nf)
    # t8, t8d = gett_all('ijll', An)#<
    t9, t9d = gett_all('ijlm', A/nf, B/nf)

    f1 = P / (P - 2)
    f2 = 2 / (P - 2)
    f3 = (1/(P-1))*(1/(P-2))

    # Denominator
    # Naive estimate
    denom_n = t1 - 2/P * t5 + (1/P)**2 * t9

    # Kong-Valiant estimate
    denom_s = P/(P-3) * (
        t1
        - f1 * t2
        + f2 * (2*t3 - t5)
        + f3 * (t6 - 2*t7 + t9)
    )

    # Our estimate
    denom_d = (P/(P-3))*(Q/(Q-1)) * (
        t1d
        - f1 * t2d
        + f2 * (2*t3d - t5d)
        + f3 * (t6d - 2*t7d + t9d)
    )

    # Numerator
    # Naive estimate
    numer_n = t6 - 2/P * t7 + (1/P)**2 * t9

    # Kong-Valiant estimate
    numer_s = P/(P-3) * (
        t6
        - 2/(P-1) * t7
        + 1/(P-2) * (4*t3 - P*t2)
        + f3 * (t9 - 4*t5 + 2*t1 - t6)
    )

    # Our estimate
    numer_d = (P/(P-3))*(Q/(Q-1))*(
        t6d
        - 2/(P-1) * t7d
        + 1/(P-2) * (4*t3d - P*t2d)
        + f3 * (t9d - 4*t5d + 2*t1d - t6d)
    )

    # Kong-Valiant estimate - column corrected
    numer_s_col = t6d - 2/P * t7d + (1/P)**2 * t9d
    denom_s_col = t1d - 2/P * t5d + (1/P)**2 * t9d

    out = [[numer_n, denom_n],
           [numer_s, denom_s],
           [numer_s_col, denom_s_col],
           [numer_d, denom_d]]
    out = [torch.stack(r) for r in out]

    return torch.stack(out)  # (4, 2)


@torch.compile(mode="max-autotune", fullgraph=False, dynamic=True)
def estimate_dimensionality_no_centering(A: torch.Tensor, B: torch.Tensor):
    """
    Same as above but w/o centering; returns (4,2) tensor.
    """
    P, Q = A.shape
    nf = (P * Q) ** 0.5

    An = A / nf
    Bn = B / nf

    t1, t1d = gett_all('iijj', An, Bn)
    t3, t3d = gett_all('ijij', An, Bn)
    t4, t4d = gett_all('iiii', An, Bn)

    numer_n = t1
    numer_s = P/(P-1) * (t1 - t4)
    numer_s_col = Q/(Q-1) * t1d
    numer_d = P/(P-1) * Q/(Q-1) * (t1d - t4d)

    denom_n = t3
    denom_s = P/(P-1) * (t3 - t4)
    denom_s_col = Q/(Q-1) * t3d
    denom_d = P/(P-1) * Q/(Q-1) * (t3d - t4d)

    out = [[numer_n, denom_n],
           [numer_s, denom_s],
           [numer_s_col, denom_s_col],
           [numer_d, denom_d]]
    out = [torch.stack(r) for r in out]

    return torch.stack(out)  # (4, 2)


def get_dimensionality(A, B, P: int, Q: int):
    """
    Estimate the dimensionality of the population A.
    B is either the same as A or a different trial of the same population.
    Returns np.ndarray with shape (4,2).
    """
    A = torch.as_tensor(A, dtype=DTYPE, device=DEVICE)
    B = torch.as_tensor(B, dtype=DTYPE, device=DEVICE)
    assert A.shape == B.shape, "A and B must have the same dimensions"
    assert A.shape == (P, Q), f"A.shape != (P, Q), {A.shape} != ({P}, {Q})"

    returns = estimate_dimensionality(A, B).cpu()
    returns = np.array(returns)

    return returns


def get_dimensionality_avg(Phi, P_ratio, Q_ratio, numit, vectorize=False):

    if Phi.ndim == 2:
        Phi = np.broadcast_to(Phi, (2, *Phi.shape))

    T_tot, P_tot, Q_tot = Phi.shape
    P = max(4, int(P_tot*P_ratio))
    Q = max(4, int(Q_tot*Q_ratio))

    Phi = torch.from_numpy(Phi.copy()).to(dtype=DTYPE, device=DEVICE)

    if vectorize:
        # def single_trial(key):
        #     g = torch.Generator(device=DEVICE).manual_seed(key[0])
        #     idx_T = torch.randint(low=0, high=T_tot, size=(2,), generator=g, device=DEVICE)
        #     idx_P = torch.randperm(P_tot, generator=g, device=DEVICE)[:P]
        #     idx_Q = torch.randperm(Q_tot, generator=g, device=DEVICE)[:Q]

        #     Phi_a = Phi[idx_T[0]][idx_P, :][:, idx_Q]
        #     Phi_b = Phi[idx_T[1]][idx_P, :][:, idx_Q]
        #     return estimate_dimensionality(Phi_a, Phi_b)

        # keys = torch.arange(numit, device=DEVICE, dtype=torch.int64)

        # try:
        #     # Use vectorized operations for better GPU utilization
        #     Ms = vmap(single_trial)(keys)
        #     return Ms
        # except Exception as e:
        #     print(f"Vectorized computation failed: {e}")
        #     # Fallback to sequential computation

        # 1) Hoist RNG: make ALL indices in batch up front with a single generator
        g = torch.Generator(device=DEVICE).manual_seed(12345)

        # Two T indices per trial: shape (B, 2)
        idx_T = torch.randint(T_tot, (numit, 2), generator=g, device=DEVICE)

        # Batched random subsets for P and Q:
        # Use random scores + topk to simulate randperm per batch
        scoresP = torch.rand((numit, P_tot), generator=g, device=DEVICE, dtype=DTYPE)
        scoresQ = torch.rand((numit, Q_tot), generator=g, device=DEVICE, dtype=DTYPE)
        idx_P = scoresP.topk(P, dim=1).indices         # (B, P)
        idx_Q = scoresQ.topk(Q, dim=1).indices         # (B, Q)

        # 2) Pure per-trial function (no RNG, no .item())
        def single_trial(idx_T_row, idx_P_row, idx_Q_row):
            # # idx_T_row: (2,), idx_P_row: (P,), idx_Q_row: (Q,)
            # t0, t1 = idx_T_row[0], idx_T_row[1]
            # # Advanced indexing per batch element:
            # Phi_a = Phi[t0][idx_P_row][:, idx_Q_row]   # (P, Q)
            # Phi_b = Phi[t1][idx_P_row][:, idx_Q_row]   # (P, Q)

            t0 = idx_T_row[0].to(torch.long)
            t1 = idx_T_row[1].to(torch.long)
            idx_P_row = idx_P_row.to(torch.long)
            idx_Q_row = idx_Q_row.to(torch.long)

            # pick T without converting to Python int
            Phi_t0 = torch.index_select(Phi, 0, t0.unsqueeze(0)).squeeze(0)     # (P_tot, Q_tot)
            Phi_t1 = torch.index_select(Phi, 0, t1.unsqueeze(0)).squeeze(0)     # (P_tot, Q_tot)

            # then select P, then Q, all with tensor indices
            Phi_a = torch.index_select(Phi_t0, 0, idx_P_row)                    # (P, Q_tot)
            Phi_a = torch.index_select(Phi_a, 1, idx_Q_row)                     # (P, Q)

            Phi_b = torch.index_select(Phi_t1, 0, idx_P_row)                    # (P, Q_tot)
            Phi_b = torch.index_select(Phi_b, 1, idx_Q_row)                     # (P, Q)

            return estimate_dimensionality(Phi_a, Phi_b)

        try:
            # Use vectorized operations for better GPU utilization
            Ms = vmap(single_trial)(idx_T, idx_P, idx_Q).cpu()
            return Ms
        except Exception as e:
            print(f"Vectorized computation failed: {e}")
            # Fallback to sequential computation

    Ms = []
    for i in range(numit):
        np.random.seed(i)
        idx_T = np.random.randint(0, T_tot, 2)
        idx_P = np.random.permutation(P_tot)[:P]
        idx_Q = np.random.permutation(Q_tot)[:Q]

        Phi_a = Phi[idx_T[0]][idx_P, :][:, idx_Q]
        Phi_b = Phi[idx_T[1]][idx_P, :][:, idx_Q]
        Ms.append(get_dimensionality(Phi_a, Phi_b, P, Q))
    Ms = np.moveaxis(Ms, 0, -1)

    return Ms
