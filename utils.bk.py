import numpy as np


def get_random_hermite_matrix(n):
    A = np.random.rand(n, n)
    A = (A + A.T) / 2
    return A


def get_random_matrix(n):
    return np.random.rand(n, n)


def power_iteration(A, k):
    lambdak = []
    vk = np.random.rand(A.shape[0])
    for i in range(k):
        vk = A @ vk
        vk = vk / np.linalg.norm(vk)
        lambdak.append(vk @ A @ vk)
    return vk, lambdak


def get_lanczos_0(A):
    v0 = np.random.rand(A.shape[0])
    v0 = v0 / np.linalg.norm(v0)
    alpha0 = v0.conj().T @ A @ v0
    beta0 = np.linalg.norm(A @ v0 - alpha0 * v0)
    return alpha0, beta0, v0


def get_lanczos_1(A, v0, alpha0, beta0):
    v1 = (A @ v0 - alpha0 * v0) / beta0
    alpha1 = v1.conj().T @ A @ v1
    beta1 = np.linalg.norm(A @ v1 - alpha1 * v1 - beta0 * v0)
    return alpha1, beta1, v1


def get_lanczos_greater_than_1(A, i, vs, alphas, betas):
    vi = (A @ vs[i - 1] - alphas[i - 1] * vs[i - 1] - betas[i - 2] * vs[i - 2]) / betas[
        i - 1
    ]
    alphai = vi.conj().T @ A @ vi
    betai = np.linalg.norm(A @ vi - alphai * vi - betas[i - 1] * vs[i - 1])
    return alphai, betai, vi


def get_alpha_beta_v(A, n):
    alphas = np.zeros(n)
    betas = np.zeros(n)
    vs = np.zeros((n, A.shape[0]))
    for i in range(n):
        if i == 0:
            alphas[0], betas[0], vs[0, :] = get_lanczos_0(A)
        elif i == 1:
            alphas[1], betas[1], vs[1, :] = get_lanczos_1(
                A, vs[0, :], alphas[0], betas[0]
            )
        else:
            alphas[i], betas[i], vs[i, :] = get_lanczos_greater_than_1(
                A, i, vs, alphas, betas
            )
    return alphas, betas, vs


def get_tridiagonal_matrix(alphas, betas):
    T = np.zeros((len(alphas), len(alphas)))
    for i in range(len(alphas)):
        T[i, i] = alphas[i]
        if i < len(alphas) - 1:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    return T
