import numpy as np
import scipy.stats as st
from scipy.integrate import quad, nquad
from functools import partial


def prior_pdf(xs, q, ax, mx, base_dist):
    """
    Construct a prior for a univariate forecast synthesis
    :param xs: observations
    :param q: probability assigned to agent
    :param ax: synthesis function
    :param mx: expected density of agent forecast
    :param base_dist: base forecast pdf
    :return:
    """
    c_f = lambda x: mx.pdf(x) * ax(x)
    c = quad(c_f, -5, 5)[0]
    p = ax(xs) * mx.pdf(xs) / c
    return (1-q*c) * base_dist.pdf(xs) + q * c * p


def synthesise(xs, q, f, s, ax, base_dist):
    """
    Synthesise the base decision maker pdf with agent pdf
    :param xs: observations
    :param q: probability assigned to agent
    :param f: mean of agent forecast
    :param s: scale of agent forecast
    :param ax: synthesis function
    :param base_dist: base forecast pdf
    :return:
    """
    agent = st.norm(loc=f, scale=s)
    cH_x = lambda x: ax(x) * agent.pdf(x)
    cH = quad(cH_x, -5, 5)[0]
    p_yH = lambda x: ax(x) * agent.pdf(x) / cH
    return q * cH * p_yH(xs) + (1 - q * cH) * base_dist.pdf(xs)


def alpha(mu_j, var_j, cor_j, mu_mj, r1, r2, d, x_j, x_mj):
    e_j = x_j - mu_j - cor_j * (x_mj - mu_mj)
    return np.exp(-e_j**2 / (2 * r1 * var_j)) - d * np.exp(-e_j**2 / (2 * r2 * var_j))


def synthesis_fun(alpha, h_mj, qj, x_j):
    alp = partial(alpha, x_j)
    fun = lambda x: alp(x) * h_mj.pdf(x)
    return qj * quad(fun, -10, 10)[0]


def a0_fun(axs, y):
    return 1 - np.sum([a(y) for a in axs])


def synthesise_2_agents(cor, r_1, r_2, d, agent1, agent2, q1, q2):
    def alpha(mu_j, var_j, cor_j, mu_mj, r1, r2, d, x_j, x_mj):
        e_j = x_j - mu_j - cor_j * (x_mj - mu_mj)
        return np.exp(-e_j ** 2 / (2 * r1 * var_j)) - d * np.exp(-e_j ** 2 / (2 * r2 * var_j))

    alpha1 = partial(alpha, 0, 1, cor, 0, r_1, r_2, d)
    alpha2 = partial(alpha, 0, 1, cor, 0, r_1, r_2, d)
    a1 = partial(synthesis_fun, alpha1, agent2, q1)
    a2 = partial(synthesis_fun, alpha2, agent1, q2)

    def a0_fun(x, y):
        return (1 - q1 * alpha(0, 1, cor, 0, r_1, r_2, 1, x, y) - q2 * alpha(0, 1, cor, 0, r_1, r_2, 1, y,
                                                                             x)) * agent1.pdf(x) * agent2.pdf(y)

    a0 = nquad(a0_fun, [[-6, 6], [-6, 6]])[0]

    mix = lambda x: a0 * st.norm.pdf(x) + a1(x) * agent1.pdf(x) + a2(x) * agent2.pdf(x)
    return mix