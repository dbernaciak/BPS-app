from .synthesis import *


def plot(axis, xs, q, ax, mx, pi_0, f, s):
    axis.plot(xs, prior_pdf(xs, q, ax, mx, pi_0), c="tab:blue", label="prior")
    axis.plot(xs, synthesise(xs, q, f, s, ax, pi_0), c="tab:red", label="posterior")
    axis.plot(xs, mx.pdf(xs), c="tab:orange", linestyle="--", label="m(y)")
    axis.plot(xs, st.norm(loc=f, scale=s).pdf(xs), c="tab:purple", linestyle="--", label="agent h(y)")
    axis.legend()


def plot_2_agents(axis, xs, mix, agent1, agent2):
    axis.plot(xs, st.norm.pdf(xs), c="tab:blue", label="prior")
    axis.plot(xs, np.asarray([mix(x) for x in xs]), c="tab:red", label="posterior")
    axis.plot(xs, agent1.pdf(xs), c="tab:orange", linestyle="--", label="agent 1")
    axis.plot(xs, agent2.pdf(xs), c="tab:purple", linestyle="--", label="agent 2")
    axis.legend()


def plot_2_agents_fast(axis, xs, mix, agent1, agent2):
    axis.plot(xs, st.norm.pdf(xs), c="tab:blue", label="prior")
    axis.plot(xs, mix, c="tab:red", label="posterior")
    axis.plot(xs, agent1.pdf(xs), c="tab:orange", linestyle="--", label="agent 1")
    axis.plot(xs, agent2.pdf(xs), c="tab:purple", linestyle="--", label="agent 2")
    axis.legend()