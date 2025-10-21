import torch


def compute_self_overlap(λ, λ_tilde, q, eta=1e-15, max_iter=10000, tol=1e-6):

    P = len(λ_tilde)
    λ = λ.to(torch.float64)
    λ_tilde = λ_tilde.to(torch.float64)

    stj_z = solve_stieltjes(λ, λ_tilde, q, eta=eta, max_iter=max_iter, tol=tol)
    hil_E = torch.real(stj_z)
    rho_E = torch.imag(stj_z) / torch.pi

    Mu = λ_tilde[None, :]
    Lambda = λ[:, None]
    hil_E = hil_E[:, None]
    rho_E = rho_E[:, None]

    numerator = q * Mu * Lambda
    denominator_real = Mu * (1 - q) - Lambda + q * Mu * Lambda * hil_E
    denominator_imag = q * Mu * Lambda * torch.pi * rho_E
    denominator = denominator_real**2 + denominator_imag**2
    Q_th = numerator / denominator
    Q_th /= P  # We divide by P to obtain the matrix Q_{ij}

    return Q_th


def solve_stieltjes(λ, λ_tilde, q, eta=1e-15, max_iter=10000, tol=1e-6):
    """
    Solves for g(z) given the equation:
    g(z) = (1/P) * sum_{i=1}^P \frac{1}{z - λ_tilde*(1 - q + q*z*g(z))} for z = λ - 1j*eta

    Parameters:
    - λ: Array of sample eigenvalues to evaluate the Stieltjes transform at.
    - λ_tilde: Array of population eigenvalues size P.
    - q (int): Aspect ratio P/Q.
    - eta (float): Small imaginary part to avoid singularities.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Tolerance for convergence.

    Returns:
    - g_z (complex or array): The computed value of g(z) for each z.
    """

    λ_tilde = λ_tilde.to(torch.float64)[None, :]
    λ = λ.to(torch.float64)[:, None]

    z = λ - 1j * eta
    g_z = torch.ones_like(z, dtype=torch.complex128)  # Starting with 1.0 as complex numbers

    for n in range(max_iter):

        denominator = z - λ_tilde * (1 - q + q * z * g_z)
        g_z_new = (1.0 / denominator).mean(axis=-1, keepdim=True)

        if torch.all(torch.abs(g_z_new - g_z) < tol):
            return g_z_new.squeeze()
        else:
            g_z = g_z_new

    return g_z_new.squeeze()


def solve_silverstein(λ, λ_tilde, q, eta=1e-15, max_iter=10000, tol=1e-6):
    """
    Solves for κ given the equation:
    κ = -z + κ * q*λ_tilde / (λ_tilde + κ)

    Parameters:
    - λ: Array of sample eigenvalues to evaluate the Silverstein transform at.
    - λ_tilde: Array of population eigenvalues size P.
    - q (int): Aspect ratio P/Q.
    - eta (float): Small imaginary part to avoid singularities.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Tolerance for convergence.

    Returns:
    - κ (complex or array): The computed value of κ for each z.
    """

    λ_tilde = λ_tilde.to(torch.float64)[None, :]
    λ = λ.to(torch.float64)[:, None]

    z = λ - 1j * eta
    κ = torch.ones_like(z, dtype=torch.complex128)  # Starting with 1.0 as complex numbers

    for n in range(max_iter):

        κ_new = -z + κ * q*λ_tilde / (λ_tilde + κ)
        κ_new = κ_new.mean(axis=-1, keepdim=True)

        if torch.all(torch.abs(κ - κ_new) < tol):
            return κ_new.squeeze()
        else:
            κ = κ_new

    return κ.squeeze()
