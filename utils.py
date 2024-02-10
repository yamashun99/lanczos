import numpy as np


def get_random_matrix(n, hermite=False):
    """
    n x nのランダム行列を生成します。hermiteがTrueの場合、行列はエルミートになります。

    パラメータ:
    - n (int): 正方行列の次元。
    - hermite (bool): Trueの場合、エルミート行列を生成します。

    戻り値:
    - A (ndarray): 生成された行列。
    """
    A = np.random.rand(n, n)
    if hermite:
        A = (A + A.T) / 2
    return A


def power_iteration(A, k):
    """
    行列Aに対してkステップのべき乗法を実行し、優位な固有値と固有ベクトルを近似します。

    パラメータ:
    - A (ndarray): 入力行列。
    - k (int): 反復回数。

    戻り値:
    - vk (ndarray): 近似された固有ベクトル。
    - lambdak (list): 近似された固有値の履歴。
    """
    lambdak = []
    vk = np.random.rand(A.shape[0])
    for _ in range(k):
        vk = A @ vk
        vk /= np.linalg.norm(vk)
        lambdak.append(vk @ A @ vk)
    return vk, lambdak


def lanczos_iteration(A, n):
    """
    nステップのランチョス反復を実行し、行列Aを三重対角行列に変換します。

    パラメータ:
    - A (ndarray): 入力行列。
    - n (int): ランチョスステップの数。

    戻り値:
    - alphas (ndarray): 三重対角行列の対角要素。
    - betas (ndarray): 三重対角行列の対角線上の要素。
    - vs (ndarray): 反復中に生成された正規直交ベクトル。
    """
    alphas = np.zeros(n)
    betas = np.zeros(n - 1)
    vs = np.zeros((n, A.shape[0]))

    v = np.random.rand(A.shape[0])
    v /= np.linalg.norm(v)
    vs[0] = v

    for i in range(n):
        w = A @ vs[i]
        alpha = np.conj(vs[i]).T @ w
        alphas[i] = alpha
        if i < n - 1:
            w -= alpha * vs[i] + (betas[i - 1] * vs[i - 1] if i > 0 else 0)
            beta = np.linalg.norm(w)
            betas[i] = beta
            vs[i + 1] = w / beta

    return alphas, betas, vs


def get_tridiagonal_matrix(alphas, betas):
    """
    与えられたalphas（対角要素）とbetas（対角線上の要素）から三重対角行列を構築します。

    パラメータ:
    - alphas (ndarray): 対角要素。
    - betas (ndarray): 対角線上の要素。

    戻り値:
    - T (ndarray): 三重対角行列。
    """
    n = len(alphas)
    T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    return T
