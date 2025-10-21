import torch
torch.set_default_dtype(torch.float64)


def gett_all(ijlm: str, A, B, indep_cols=True):
    i, j, l, m = list(ijlm)
    pexp = i+'a,'+j+'a,'+l+'b,'+m+'b->'
    qexp = i+'a,'+j+'a,'+l+'a,'+m+'a->'
    pval = torch.einsum(pexp, A, A, B, B)
    if indep_cols or A.shape != B.shape:
        pqval = pval*0
    else:
        pqval = pval - torch.einsum(qexp, A, A, B, B)
    return pval, pqval


# @torch.compile(mode="max-autotune", fullgraph=False, dynamic=True)
def estimate_cka_moments(A, B, indep_cols=True):

    P, Qa = A.shape
    P, Qb = B.shape
    nf_a = (P * Qa) ** 0.5
    nf_b = (P * Qb) ** 0.5

    t1, t1d = gett_all('ijji', A/nf_a, B/nf_b, indep_cols=indep_cols)
    t2, t2d = gett_all('iiii', A/nf_a, B/nf_b, indep_cols=indep_cols)
    t3, t3d = gett_all('ijjj', A/nf_a, B/nf_b, indep_cols=indep_cols)
    t4, t4d = gett_all('iiij', A/nf_a, B/nf_b, indep_cols=indep_cols)
    t5, t5d = gett_all('ijjl', A/nf_a, B/nf_b, indep_cols=indep_cols)
    t6, t6d = gett_all('iijj', A/nf_a, B/nf_b, indep_cols=indep_cols)
    t7, t7d = gett_all('iijl', A/nf_a, B/nf_b, indep_cols=indep_cols)
    t8, t8d = gett_all('ijll', A/nf_a, B/nf_b, indep_cols=indep_cols)
    t9, t9d = gett_all('ijlm', A/nf_a, B/nf_b, indep_cols=indep_cols)

    f1 = P / (P - 2)
    f2 = 2 / (P - 2)
    f3 = (1/(P-1))*(1/(P-2))

    # Naive estimate
    sums_n = t1 - 2/P * t5 + (1/P)**2 * t9

    # Kong-Valiant estimate
    sums = P/(P-3) * (
        t1
        - f1 * t2
        + f2 * (t3 + t4 - t5)
        + f3 * (t6 - t7 - t8 + t9)
    )

    # Our estimate - (Q = Qa = Qb  # Otherwise, the einsum code will throw an error.)
    sums_d = (P/(P-3))*(Qa/(Qa-1)) * (
        t1d
        - f1 * t2d
        + f2 * (t3d + t4d - t5d)
        + f3 * (t6d - t7d - t8d + t9d)
    )

    if indep_cols or A.shape != B.shape:
        sums_d = sums

    return torch.stack([sums_n, sums_d])
