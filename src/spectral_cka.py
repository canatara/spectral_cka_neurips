import numpy as np
import torch
from .rmt_utils import compute_self_overlap
from .cka_utils import estimate_cka_moments


CCA_THRESHOLD = 10


def spectrum_gram_matrix(G):

    # Compute the eigenvalues and eigenvectors of the Gram matrix
    assert G.shape[0] == G.shape[1], "The Gram matrix must be square"
    G = G.to(torch.float64)
    P = len(G)

    eigs, vecs = torch.linalg.eigh(G / P)
    sort_idx = torch.argsort(eigs, descending=True)
    eigs = eigs[sort_idx]
    vecs = vecs[:, sort_idx] * np.sqrt(P)

    return eigs, vecs


def compute_overlap(vecs1, vecs2):

    P = len(vecs1)
    M_tilde = vecs1.T @ vecs2 / P
    M_tilde = (M_tilde**2)  # M_{ij}

    return M_tilde


def compute_cka(eigs1, eigs2, overlap):

    cka_numerator = eigs1 @ overlap @ eigs2
    cka_denominator = torch.sqrt((eigs1**2).sum() * (eigs2**2).sum())
    return cka_numerator / cka_denominator


def predict_cka(act1, act2, neuron_sizes, cutoff=10, num_iterations=20, use_real_neuron=False):
    """
    Compute the predicted CKA from population activations. act1 and act2 are the population activations.
    act2 is kept deterministic and only act1 is neuron subsampled.

    neuron_sizes: list of neuron sizes to subsample from act1
    cutoff: number of eigenvalues to consider
    num_iterations: number of random neuron samplings to average over
    use_real_neuron: whether to subsample from real neurons or generate random neurons
    """
    assert act1.shape[0] == act2.shape[0], "The number of samples must be the same"

    act1 = act1.to(torch.float64) / np.sqrt(act1.shape[1])
    act2 = act2.to(torch.float64) / np.sqrt(act2.shape[1])

    # Compute eigenvalues and eigenvectors of act1 and act2
    gram1 = act1 @ act1.T
    eigs1, vecs1 = spectrum_gram_matrix(gram1)

    gram2 = act2 @ act2.T
    eigs2, vecs2 = spectrum_gram_matrix(gram2)

    # Compute the population overlap matrix
    M_tilde = compute_overlap(vecs1, vecs2)

    # compute true CKA(computed up to 10 eigenvalues)
    P, N1, N2 = act1.shape[0], act1.shape[1], act2.shape[1]
    true_cka = compute_cka(eigs1, eigs2, M_tilde)
    true_cca = torch.sum(M_tilde[:CCA_THRESHOLD, :CCA_THRESHOLD]) / CCA_THRESHOLD
    print(f"P={P}, N1={N1}, N2={N2}, True CKA: {true_cka}, True CCA: {true_cca}")

    computed_results = {'true_cka': true_cka.cpu(), 'true_cca': true_cca.cpu()}
    for neuron_size in neuron_sizes:
        print(f"Neuron_size={neuron_size}/{N1}")
        returns = compute_and_average_matrices(eigs1, vecs1,
                                               eigs2, vecs2,
                                               act1, act2,
                                               M_tilde, neuron_size,
                                               cutoff=cutoff,
                                               num_iterations=num_iterations,
                                               use_real_neuron=use_real_neuron)

        for key, val in returns.items():
            if computed_results.get(key) is None:
                computed_results[key] = []
            computed_results[key].append(val)

    return computed_results


def compute_and_average_matrices(eigs1, vecs1,
                                 eigs2, vecs2,
                                 act1, act2,
                                 M_tilde, neuron_size,
                                 cutoff=None,
                                 num_iterations=20,
                                 use_real_neuron=False):

    P, N_tilde = act1.shape
    N = neuron_size

    eigs_trial = []
    Q_th_trial = []
    Q_emp_trial = []
    M_emp_trial = []
    M_pred_trial = []
    M_tilde_pred_trial = []

    cka_naive_all = []
    cka_pred_all = []
    cka_est_all = []

    cca_naive_all = []
    cca_pred_all = []
    cca_est_all = []

    moment_cka_est_all = []

    for trial in range(num_iterations):

        if use_real_neuron:
            # Randomly subsample N neurons from population activations
            idx = torch.randperm(N_tilde)[:N]
            sample_act1 = act1[:, idx] * np.sqrt(N_tilde/N)
        else:
            # Generate a (N_tilde x N) random projection to sample from population activations
            R = torch.randn((N_tilde, N), dtype=act1.dtype, device=act1.device) / np.sqrt(N)
            sample_act1 = act1 @ R

        # Sample eigenvalues and eigenvectors from (P x P)-sample Gram matrix
        sample_gram1 = sample_act1 @ sample_act1.T
        s_eigs1, s_vecs1 = spectrum_gram_matrix(sample_gram1)

        # Calculate the overlap matrix and slice it according to the ranks
        Q_emp = compute_overlap(s_vecs1[:, :], vecs1[:, :])
        M_emp = compute_overlap(s_vecs1[:, :], vecs2[:, :])

        # Compute theoretical CKA
        (M_pred,
         Q_th,
         eig_ratio,
         naive_cka,
         predicted_cka,
         estimated_cka,
         naive_cca,
         predicted_cca,
         estimated_cca,
         M_tilde_pred) = theoretical_cka(eigs1, s_eigs1,
                                         eigs2, M_emp, M_tilde,
                                         N, cutoff)

        # Average empirical eigenvalues and matrices
        eigs_trial.append(s_eigs1)
        Q_th_trial.append(Q_th)
        Q_emp_trial.append(Q_emp)
        M_emp_trial.append(M_emp)
        M_pred_trial.append(M_pred)
        M_tilde_pred_trial.append(M_tilde_pred)

        cka_naive_all.append(naive_cka)
        cka_pred_all.append(predicted_cka)
        cka_est_all.append(estimated_cka)

        cca_naive_all.append(naive_cca)
        cca_pred_all.append(predicted_cca)
        cca_est_all.append(estimated_cca)

        num = estimate_cka_moments(sample_act1, act2, indep_cols=True).cpu()
        denom1 = estimate_cka_moments(sample_act1, sample_act1, indep_cols=False).cpu()
        denom2 = estimate_cka_moments(act2, act2, indep_cols=False).cpu()
        moment_cka_est_all.append([num, denom1, denom2])

    def unbiased_cka_and_error(moment_cka_est_all):

        x1, x2, x3 = np.stack(moment_cka_est_all).mean(axis=0)
        s1, s2, s3 = np.stack(moment_cka_est_all).std(axis=0)

        term1 = (s1**2) / (x2 * x3)
        term2 = (x1**2 * s2**2) / (4 * x2**3 * x3)
        term3 = (x1**2 * s3**2) / (4 * x3**3 * x2)
        y = x1 / np.sqrt(x2 * x3)
        sigma_y = np.sqrt(term1 + term2 + term3)

        return y, sigma_y

    moment_cka_est_mean, moment_cka_est_std = unbiased_cka_and_error(moment_cka_est_all)

    returns = dict(eigs_mean=torch.stack(eigs_trial).mean(dim=0),
                   eigs_std=torch.stack(eigs_trial).std(dim=0),

                   Q_th_mean=torch.stack(Q_th_trial).mean(dim=0),
                   Q_th_std=torch.stack(Q_th_trial).std(dim=0),

                   Q_emp_mean=torch.stack(Q_emp_trial).mean(dim=0),
                   Q_emp_std=torch.stack(Q_emp_trial).std(dim=0),

                   M_emp_mean=torch.stack(M_emp_trial).mean(dim=0),
                   M_emp_std=torch.stack(M_emp_trial).std(dim=0),

                   M_pred_mean=torch.stack(M_pred_trial).mean(dim=0),
                   M_pred_std=torch.stack(M_pred_trial).std(dim=0),

                   M_tilde_pred_mean=torch.stack(M_tilde_pred_trial).mean(dim=0),
                   M_tilde_pred_std=torch.stack(M_tilde_pred_trial).std(dim=0),

                   cka_naive_mean=torch.stack(cka_naive_all).mean(dim=0),
                   cka_naive_std=torch.stack(cka_naive_all).std(dim=0),

                   cka_pred_mean=torch.stack(cka_pred_all).mean(dim=0),
                   cka_pred_std=torch.stack(cka_pred_all).std(dim=0),

                   cka_est_mean=torch.stack(cka_est_all).mean(dim=0),
                   cka_est_std=torch.stack(cka_est_all).std(dim=0),

                   cca_naive_mean=torch.stack(cca_naive_all).mean(dim=0),
                   cca_naive_std=torch.stack(cca_naive_all).std(dim=0),

                   cca_pred_mean=torch.stack(cca_pred_all).mean(dim=0),
                   cca_pred_std=torch.stack(cca_pred_all).std(dim=0),

                   cca_est_mean=torch.stack(cca_est_all).mean(dim=0),
                   cca_est_std=torch.stack(cca_est_all).std(dim=0),

                   moment_cka_est_mean=moment_cka_est_mean,
                   moment_cka_est_std=moment_cka_est_std,
                   )

    # Compute theoretical CKA
    (M_pred,
     Q_th,
     eig_ratio,
     naive_cka,
     predicted_cka,
     estimated_cka,
     naive_cca,
     predicted_cca,
     estimated_cca,
     M_tilde_pred) = theoretical_cka(eigs1, returns['eigs_mean'],
                                     eigs2, returns['M_emp_mean'], M_tilde,
                                     N, cutoff)

    returns |= dict(M_pred_final=M_pred,
                    Q_th_final=Q_th,
                    eig_ratio_final=eig_ratio,
                    naive_cka_final=naive_cka,
                    predicted_cka_final=predicted_cka,
                    estimated_cka_final=estimated_cka,
                    naive_cca_final=naive_cca,
                    predicted_cca_final=predicted_cca,
                    estimated_cca_final=estimated_cca,
                    M_tilde_pred_final=M_tilde_pred,
                    M_tilde=M_tilde)

    returns = {key: val.cpu().numpy() if torch.is_tensor(val) else val for key, val in returns.items()}

    return returns


def theoretical_cka(eigs1, s_eigs1,
                    eigs2, M_emp, M_tilde,
                    neuron_size, cutoff=None):

    P = len(eigs1)
    λ = s_eigs1
    λ_tilde = eigs1

    # Theoretical prediction for Q_{ij} and M_{ia}
    q = P/neuron_size
    eta = 1/np.sqrt(P)
    eta = 1e-15
    Q_th = compute_self_overlap(λ[:cutoff], λ_tilde, q, eta=eta)
    M_pred = Q_th @ M_tilde

    # Estimated M_tilde from data
    M_tilde_reg = optimize_tilde_M_grad(Q_th[:cutoff], M_emp[:cutoff])
    # print(M_tilde_reg.sum(1).mean(), M_tilde_reg.sum(0).mean())
    # print(M_tilde_reg.sum(0).mean(), M_tilde_reg.sum(1).mean())

    # Naive CKA
    naive_cka = compute_cka(s_eigs1[:cutoff], eigs2, M_emp[:cutoff])

    # Predicted CKA
    predicted_cka = compute_cka(s_eigs1[:cutoff], eigs2, M_pred[:cutoff])

    # Estimated CKA
    estimated_cka = compute_cka(s_eigs1[:cutoff], eigs2, M_tilde_reg[:cutoff])

    eig_ratio = (λ/λ_tilde).mean()

    # Naive CCA
    naive_cca = M_emp[:CCA_THRESHOLD, :CCA_THRESHOLD].sum() / CCA_THRESHOLD
    predicted_cca = M_pred[:CCA_THRESHOLD, :CCA_THRESHOLD].sum() / CCA_THRESHOLD
    estimated_cca = M_tilde_reg[:CCA_THRESHOLD, :CCA_THRESHOLD].sum() / CCA_THRESHOLD

    returns = (M_pred, Q_th, eig_ratio, naive_cka,
               predicted_cka, estimated_cka, naive_cca, predicted_cca, estimated_cca, M_tilde_reg)

    returns = (item.cpu() for item in returns)

    return returns


def experiment(act1, act2,
               sample_sizes,
               neuron_sizes,
               num_trials=5,
               num_iterations=20,
               cutoff=10,
               use_real_neuron=True,
               fn_prefix='data',
               fn_suffix='',
               override=False):

    fn_suffix = f"_{fn_suffix}" if len(fn_suffix) > 0 else ""
    fn_suffix = fn_suffix + "_real_neuron" if use_real_neuron else fn_suffix

    import os
    os.makedirs('./results', exist_ok=True)

    all_data = []
    for P in sample_sizes:
        filename = f'./results/{fn_prefix}_{P}_trials_{num_trials}_iters_{num_iterations}{fn_suffix}.npz'
        if os.path.exists(filename) and not override:
            total_data = np.load(filename, allow_pickle=True)['total_data'].item()
            all_data += [total_data]
            print(f"File {filename} exists. Skipping...")

        else:
            print(f"File {filename} does not exist. Running the experiment...")
            computed_results = []
            for trial in range(num_trials):
                # Load activations
                idx = torch.randperm(act1.size(0))[:P]
                act1_sub = act1[idx]
                act2_sub = act2[idx]
                result = predict_cka(act1_sub, act2_sub,
                                     neuron_sizes,
                                     cutoff,
                                     num_iterations,
                                     use_real_neuron=use_real_neuron)
                computed_results += [result]

            total_data = {key: [] for key in result.keys()}
            for data in computed_results:
                for key, val in data.items():
                    total_data[key] += [val]
            # total_data = {key: np.asarray(val).T for key, val in total_data.items()}
            total_data = {key: np.moveaxis(np.asarray(val), 0, -1) for key, val in total_data.items()}
            np.savez(filename, total_data=total_data, sample_sizes=sample_sizes, neuron_sizes=neuron_sizes)
            all_data += [total_data]

        cka_mean = total_data['cka_naive_mean'].mean(-1)
        cka_std = total_data['cka_naive_std'].mean(-1)
        predicted_cka = total_data['predicted_cka_final'].mean(-1)
        estimated_cka = total_data['estimated_cka_final'].mean(-1)

        import matplotlib.pyplot as plt
        label_fontsize = 14
        plt.figure(figsize=(6, 4))
        plt.semilogx(neuron_sizes, cka_mean, label='Sample_CKA Empirical', color='black')
        plt.fill_between(neuron_sizes, cka_mean - cka_std, cka_mean + cka_std, color='gray', alpha=0.2)
        plt.semilogx(neuron_sizes, predicted_cka, 'x', label='Sample_CKA Theory', color='red')
        plt.semilogx(neuron_sizes, estimated_cka, '-o', label='Estimated_CKA Theory', color='blue')

        plt.xlabel(r"Neuron size $N$", fontsize=label_fontsize)
        plt.ylabel(r"CKA", fontsize=label_fontsize)

        plt.grid(True, which='both', linestyle='--', linewidth=0.3)
        # plt.xlim([9, 550])
        ymax = min(1.0, plt.gca().get_ylim()[1])*1.1
        plt.ylim([None, ymax])
        plt.legend(loc='lower right', fontsize=10)
        plt.tight_layout()
        plt.title(rf"CKA vs Neuron size when $P={P}$")
        plt.show()

    return all_data


def optimize_tilde_M_grad(Q, M, lr=1e-2, max_iter=1000, tol=1e-7, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Solve  min_{M_tilde >= 0} ||M - Q M_tilde||_F^2
    using Projected Gradient Descent with PyTorch's Adam optimizer.

    Q: (m, P) self-overlap matrix
    M: (m, P) cross-overlap matrix
    lr: learning rate (step size)
    max_iter: max number of PGD steps
    tol: early stopping tolerance on improvement
    beta1, beta2: Adam momentum parameters
    eps: Adam epsilon for numerical stability

    Returns:
      M_tilde: (P, P) estimated population overlap matrix
    """
    # Dimensions
    assert Q.shape == M.shape, f"Dimension mismatch. {Q.shape} != {M.shape}"

    # Initialize M_tilde
    M_tilde = estimate_tilde_M(M, Q)
    M_tilde.requires_grad = True

    # Create Adam optimizer
    optimizer = torch.optim.Adam([M_tilde], lr=lr, betas=(beta1, beta2), eps=eps)

    # We'll do iterative updates in a loop
    prev_loss = torch.inf

    for it in range(max_iter):

        # Compute loss = 0.5 * ||residual||^2
        residual = M - Q @ M_tilde
        loss = residual.pow(2).sum()
        # λ = 1e-2  # To impose doubly stochastic constraint
        # loss += λ*(M_tilde.sum(1) - 1.0).pow(2).sum()
        # loss += λ*(M_tilde.sum(0) - 1.0).pow(2).sum()

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Adam step
        optimizer.step()

        # Project to nonnegative constraint and apply Sinkhorn normalization
        with torch.no_grad():
            M_tilde.clamp_(min=0.0, max=1.0)  # Ensure non-negative
            # if it % 1 == 0:
            #     # Apply Sinkhorn normalization
            #     M_tilde.data = sinkhorn_log_script(M_tilde, iters=30, eps=1e-12)

        if torch.abs(prev_loss - loss) < tol:
            break
        elif it == max_iter-1:
            break
        else:
            prev_loss = loss

    print(it, torch.abs(prev_loss - loss).item())

    M_tilde.requires_grad = False
    return M_tilde


def estimate_tilde_M(M, Q, cutoff=10):

    U, S, Vh = torch.linalg.svd(Q, full_matrices=False)

    Sinv = 1 / S
    # Sinv[cutoff:] = 0

    Sq = torch.cumsum(S**2, dim=0) / (S**2).sum()
    Sinv[Sq > 0.98] = 0

    M_tilde = Vh.T @ torch.diag_embed(Sinv) @ U.T @ M
    M_tilde = torch.clamp(M_tilde, min=0.0, max=1.0)  # Non-negative constraint

    return M_tilde


def sinkhorn_log(M: torch.Tensor, iters: int = 30, eps: float = 1e-12) -> torch.Tensor:
    """Return a (numerically) doubly-stochastic copy of M (no in-place)."""
    logM = torch.log(M.clamp_min(eps))  # strictly positive

    u = torch.zeros_like(logM[:, 0])   # n
    v = torch.zeros_like(logM[0, :])   # n
    for _ in range(iters):
        # row normalization: u_i = -logsumexp_j (logM_{ij} + v_j)
        u = -torch.logsumexp(logM + v.unsqueeze(0), dim=1)
        # column normalization: v_j = -logsumexp_i (logM_{ij} + u_i)
        v = -torch.logsumexp(logM + u.unsqueeze(1), dim=0)
    logD = logM + u.unsqueeze(1) + v.unsqueeze(0)

    D = torch.exp(logD).clamp_min(eps)
    # D /= D.sum(dim=1, keepdim=True)
    # D /= D.sum(dim=0, keepdim=True)

    return D  # ~ doubly stochastic


sinkhorn_log_script = torch.jit.script(sinkhorn_log)  # first call compiles


def sinkhorn_normalize(M, tol=1e-9, max_iter=5):
    # M = M.clone()
    for _ in range(max_iter):
        M /= M.sum(dim=1, keepdim=True)  # Normalize rows in-place
        M /= M.sum(dim=0, keepdim=True)  # Normalize columns in-place
        if (torch.allclose(M.sum(dim=0), torch.ones_like(M.sum(dim=0)), atol=tol) and
                torch.allclose(M.sum(dim=1), torch.ones_like(M.sum(dim=1)), atol=tol)):
            break
    return M
