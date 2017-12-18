from scipy.stats import invwishart
from scipy.special import logit
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# GLOBAL Variables
ITS = 2000
BURN = 100
M = 5000
N1 = 10000
N2 = 10000
H1 = .50
H2 = .50
Ns = 0
A00 = .60
A10 = .10
A01 = .10
A11 = 0.20
C1 = np.empty(M)
C2 = np.empty(M)
SIGMA_GAMMA = np.empty((2, 2))
GAMMA1 = np.empty(M)
GAMMA2 = np.empty(M)
rho = 0
rho_e = 0
V = np.identity(M)


def sigmoid(x):
    # prevent out of range error
    if x < -6:
        x = -6
    return 1 / (1 + math.exp(-x))


def simulate():
    x1 = np.random.uniform(0, 1, (N1, M))
    x1 = (x1 - np.mean(x1, axis=0)) / np.std(x1, axis=0)
    x2 = np.random.uniform(0, 1, (N2, M))
    x2 = (x2 - np.mean(x2, axis=0)) / np.std(x2, axis=0)

    s_m_1 = 0
    for m in range(0, M):
        s_m_1 = s_m_1 + np.dot(x1[:, m], x1[:, m])

    sig_11 = (H1 / (s_m_1 * (A11 + A10)))
    sig_22 = (H2 / (s_m_1 * (A11 + A01)))
    sig_12 = (rho / (s_m_1 * A11))
    sig_21 = sig_12

    c = np.random.multinomial(1, [A00, A01, A10, A11], M)

    for m in range(0, M):
        if c[m, 0] == 1:
            C1[m] = 0
            C2[m] = 0
        elif c[m, 1] == 1:
            C1[m] = 1
            C2[m] = 0
        elif c[m, 2] == 1:
            C1[m] = 0
            C2[m] = 1
        else:
            C1[m] = 1
            C2[m] = 1

    mu = [0, 0]
    cov = [[sig_11, sig_12], [sig_21, sig_22]]
    sigma_gamma = np.array(cov)
    SIGMA_GAMMA[:] = sigma_gamma[:]
    gamma = np.random.multivariate_normal(mu, cov, M)
    GAMMA1[:] = gamma[:, 0]
    GAMMA2[:] = gamma[:, 1]

    beta1 = np.empty(M)
    beta2 = np.empty(M)
    for m in range(0, M):
        beta1[m] = gamma[m, 0] * (c[m, 1] + c[m, 3])
        beta2[m] = gamma[m, 1] * (c[m, 2] + c[m, 3])

    Sig_11 = (1 - H1) / N1
    Sig_22 = (1 - H2) / N2
    Sig_12 = (Ns * rho_e) / (N1 * N2)
    Sig_21 = (Ns * rho_e) / (N1 * N2)

    mu = np.concatenate((np.matmul(V, beta1), np.matmul(V, beta2)))
    cov = np.bmat([[Sig_11 * V, Sig_12 * V], [Sig_21 * V, Sig_22 * V]])
    z = np.random.multivariate_normal(mu, cov, 1)
    z = z.ravel()
    z1 = z[0:M]
    z2 = z[M:2 * M]

    return z1, z2


def initialize():
    # initialize t0: a11, a10, a01, a00
    lam = 1
    a = np.random.dirichlet([lam, lam, lam, lam], 1)
    a = a.ravel()
    a00 = a[0]
    a10 = a[1]
    a01 = a[2]
    a11 = a[3]

    # initialize t0: c1, c2
    c1 = np.random.binomial(1, 0.5, M)
    c2 = np.random.binomial(1, 0.5, M)

    # initialize t0: gammas
    gamma1 = np.random.normal(0, .01, M)
    gamma2 = np.random.normal(0, .01, M)

    # initialize t0: covariance matrix
    df0 = M
    scale0 = np.identity(2)
    sigma_gamma = invwishart.rvs(df0, scale0)

    return a00, a10, a01, a11, c1, c2, gamma1, gamma2, sigma_gamma


def draw_c(a00, a10, a01, a11, c1, c2, gamma1, gamma2, z1, z2, debug=False):
    sigma_B1 = (1 - H1) / N1
    sigma_B2 = (1 - H2) / N2

    for m in range(0, M):
        B1 = (-1 / float(2 * sigma_B1)) * (gamma1[m] * gamma1[m] - 2 * z1[m] * gamma1[m]) + logit(a11 + a10)
        B2 = (-1 / float(2 * sigma_B2)) * (gamma2[m] * gamma2[m] - 2 * z2[m] * gamma2[m]) + logit(a11 + a01)

        p1 = sigmoid(B1)
        p2 = sigmoid(B2)

        c1m = np.random.binomial(1, p1, 1)
        c2m = np.random.binomial(1, p2, 1)
        c1[m] = c1m
        c2[m] = c2m

    if debug is True:
        c1[:] = C1[:]
        c2[:] = C2[:]

    return c1, c2


def draw_gamma(c1, c2, gamma1, gamma2, sigma_gamma, z1, z2, debug=False):
    sigma_B1 = (1 - H1) / N1
    sigma_B2 = (1 - H2) / N2

    for m in range(0, M):
        sigma_gamma_11 = sigma_gamma[0, 0]
        sigma_gamma_22 = sigma_gamma[1, 1]

        # params of posterior distribution
        sigma_gamma_pos1 = 1/float((c1[m]/sigma_B1) + (1/sigma_gamma_11))
        mu_gamma_pos1 = (z1[m]*c1[m] / float(sigma_B1)) / float((c1[m]/sigma_B1) + (1/sigma_gamma_11))
        sigma_gamma_pos2 = 1/float((c2[m]/sigma_B2) + (1/sigma_gamma_22))
        mu_gamma_pos2 = (z2[m]*c2[m] / sigma_B2) / float((c2[m]/sigma_B2) + (1/sigma_gamma_22))

        gamma1[m] = np.random.normal(mu_gamma_pos1, math.sqrt(sigma_gamma_pos1), 1)
        gamma2[m] = np.random.normal(mu_gamma_pos2, math.sqrt(sigma_gamma_pos2), 1)

    if debug is True:
        gamma1[:] = GAMMA1[:]
        gamma2[:] = GAMMA2[:]

    return gamma1, gamma2


def draw_sigma_gamma(gamma1, gamma2, debug=False): # covariance matrix
    df0 = M
    scale0 = np.identity(2)

    gamma_two_trait = np.vstack((gamma1, gamma2))
    df = df0 + M
    psi = np.zeros(2)
    for m in range(0, M):
        psi = psi + np.matmul(gamma_two_trait[:, m], np.transpose(gamma_two_trait[:, m]))
    scale = scale0 + psi

    sigma_gamma = invwishart.rvs(df, scale)

    if debug is True:
        sigma_gamma = SIGMA_GAMMA

    return sigma_gamma


def draw_a(c1, c2, debug=False):
    lam = 1
    ones = np.ones(M)
    alpha_0 = np.sum((ones - c1) * (ones - c2)) + lam
    alpha_1 = np.sum(c1 * (ones - c2)) + lam
    alpha_2 = np.sum(c2 * (ones - c1)) + lam
    alpha_3 = np.sum(c1 * c2) + lam

    a = np.random.dirichlet([alpha_0, alpha_1, alpha_2, alpha_3], 1)
    a = a.ravel()
    a00 = a[0]
    a10 = a[1]
    a01 = a[2]
    a11 = a[3]

    if debug is True:
        a00 = A00
        a10 = A10
        a01 = A01
        a11 = A11

    return a00, a10, a01, a11


def evaluate_a(a00_t, a10_t, a01_t, a11_t):
    print np.median(a00_t[BURN:])
    print np.median(a10_t[BURN:])
    print np.median(a01_t[BURN:])
    print np.median(a11_t[BURN:])

    plt.figure()
    plt.plot(range(0, ITS), a00_t)
    plt.axhline(A00, color='r', linestyle='dashed', linewidth=2)
    plt.title("Prob of causal SNP - A00")
    plt.savefig('a00_plot.png')

    plt.figure()
    plt.plot(range(0, ITS), a10_t)
    plt.axhline(A10, color='r', linestyle='dashed', linewidth=2)
    plt.title("Prob of causal SNP - A10")
    plt.savefig('a10_plot.png')

    plt.figure()
    plt.plot(range(0, ITS), a01_t)
    plt.axhline(A01, color='r', linestyle='dashed', linewidth=2)
    plt.title("Prob of causal SNP - A01")
    plt.savefig('a01_plot.png')

    plt.figure()
    plt.plot(range(0, ITS), a11_t)
    plt.axhline(A11, color='r', linestyle='dashed', linewidth=2)
    plt.title("Prob of causal SNP - A11")
    plt.savefig('a11_plot.png')

    med00 = np.median(a00_t[BURN:])
    var00 = np.var(a00_t[BURN:])
    plt.figure()
    plt.hist(a00_t[BURN:], normed=True)
    plt.axvline(A00, color='r', linestyle='dashed', linewidth=2)
    plt.title("Prob of causal SNP - A00: med=%f, var=%f" % (med00, var00))
    plt.savefig('a00_hist.png')

    med10 = np.median(a10_t[BURN:])
    var10 = np.var(a10_t[BURN:])
    plt.figure()
    plt.hist(a10_t[BURN:], normed=True)
    plt.axvline(A10, color='r', linestyle='dashed', linewidth=2)
    plt.title("Prob of causal SNP - A10: med=%f, var=%f" % (med10, var10))
    plt.savefig('a10_hist.png')

    med01 = np.median(a01_t[BURN:])
    var01 = np.var(a01_t[BURN:])
    plt.figure()
    plt.hist(a01_t[BURN:], normed=True)
    plt.axvline(A01, color='r', linestyle='dashed', linewidth=2)
    plt.title("Prob of causal SNP - A01: med=%f, var=%f" % (med01, var01))
    plt.savefig('a01_hist.png')

    med11 = np.median(a11_t[BURN:])
    var11 = np.median(a11_t[BURN:])
    plt.figure()
    plt.hist(a11_t[BURN:], normed=True)
    plt.axvline(A11, color='r', linestyle='dashed', linewidth=2)
    plt.title("Prob of causal SNP - A11: med=%f, var=%f" % (med11, var11))
    plt.savefig('a11_hist.png')


def evaluate_gamma(gamma1_t, gamma2_t):

    gamma1_med = np.median(gamma1_t, axis=0)
    print "Estimate: %f, Truth: %f" % (gamma1_med[0], GAMMA1[0])
    print "Estimate: %f, Truth: %f" % (gamma1_med[1], GAMMA1[1])
    print "Estimate: %f, Truth: %f" % (gamma1_med[2], GAMMA1[2])
    print "Estimate: %f, Truth: %f" % (gamma1_med[3], GAMMA1[3])
    print "Estimate: %f, Truth: %f" % (gamma1_med[4], GAMMA1[4])

    gamma1_t_mat = np.matrix(gamma1_t)

    plt.figure()
    plt.plot(range(0, ITS), gamma1_t_mat[:, 0])
    plt.axhline(GAMMA1[0], color='r', linestyle='dashed', linewidth=2)
    plt.title("Gamma effect size - SNP1")
    plt.savefig('gamma1_snp1_plot.png')

    plt.figure()
    plt.plot(range(0, ITS), gamma1_t_mat[:, 1])
    plt.axhline(GAMMA1[1], color='r', linestyle='dashed', linewidth=2)
    plt.title("Gamma effect size - SNP2")
    plt.savefig('gamma1_snp2_plot.png')

    plt.figure()
    plt.plot(range(0, ITS), gamma1_t_mat[:, 2])
    plt.axhline(GAMMA1[2], color='r', linestyle='dashed', linewidth=2)
    plt.title("Gamma effect size - SNP3")
    plt.savefig('gamma1_snp3_plot.png')

    med1 = np.median(gamma1_t_mat[BURN, 0])
    var1 = np.var(gamma1_t_mat[BURN, 0])
    plt.figure()
    plt.hist(gamma1_t_mat[BURN, 0], normed=True, bins=10)
    plt.axvline(GAMMA1[0], color='r', linestyle='dashed', linewidth=2)
    plt.title("Gamma effect size - SNP 1: med=%f, var=%f" % (med1, var1))
    plt.savefig('gamma1_snp1_hist.png')

    med2 = np.median(gamma1_t_mat[BURN, 1])
    var2 = np.var(gamma1_t_mat[BURN, 1])
    plt.figure()
    plt.hist(gamma1_t_mat[BURN, 1], normed=True, log=True)
    plt.axvline(GAMMA1[1], color='r', linestyle='dashed', linewidth=2)
    plt.title("Gamma effect size - SNP 2: med=%f, var=%f" % (med2, var2))
    plt.savefig('gamma1_snp2_hist.png')

    med3 = np.median(gamma1_t_mat[BURN, 2])
    var3 = np.var(gamma1_t_mat[BURN, 2])
    plt.figure()
    plt.hist(gamma1_t_mat[BURN, 2], normed=True, log=True)
    plt.axvline(GAMMA1[2], color='r', linestyle='dashed', linewidth=2)
    plt.title("Gamma effect size - SNP 3: med=%f, var=%f" % (med3, var3))
    plt.savefig('gamma1_snp3_hist.png')


def evaluate_sigma_gamma(sigma_gamma_11_t, sigma_gamma_22_t):
    sigma_gamma_11_med = np.median(sigma_gamma_11_t)
    sigma_gamma_22_med = np.median(sigma_gamma_22_t)

    print "Estimate sig11: %f, Truth: %f" % (sigma_gamma_11_med, SIGMA_GAMMA[0, 0])
    print "Estimate sig22: %f, Truth: %f" % (sigma_gamma_22_med, SIGMA_GAMMA[1, 1])

    plt.figure()
    plt.axhline(SIGMA_GAMMA[0,0], color='r', linestyle='dashed', linewidth=2)
    plt.plot(range(0, ITS), sigma_gamma_11_t)
    plt.title("Covariance matrix - Trait 1")
    plt.savefig('sig_gamma11_plot.png')

    plt.figure()
    plt.axhline(SIGMA_GAMMA[1, 1], color='r', linestyle='dashed', linewidth=2)
    plt.plot(range(0, ITS), sigma_gamma_22_t)
    plt.title("Covariance matrix - Trait 2")
    plt.savefig('sig_gamma22_plot.png')

    med11 = np.median(sigma_gamma_11_t)
    var11 = np.var(sigma_gamma_11_t)
    plt.figure()
    plt.hist(sigma_gamma_11_t[BURN:])
    plt.axvline(SIGMA_GAMMA[0, 0], color='r', linestyle='dashed', linewidth=2)
    plt.title("Covariance matrix - Trait 1: med=%f, var=%f" % (med11, var11))
    plt.savefig('sig_gamma11_hist.png')

    med22 = np.median(sigma_gamma_22_t)
    var22 = np.median(sigma_gamma_22_t)
    plt.figure()
    plt.hist(sigma_gamma_22_t[BURN:])
    plt.axvline(SIGMA_GAMMA[1, 1], color='r', linestyle='dashed', linewidth=2)
    plt.title("Covariance matrix - Trait 2: med=%f, var=%f" % (med22, var22))
    plt.savefig('sig_gamma22_hist.png')


def main():

    random.seed(10)

    a00_t = []
    a10_t = []
    a01_t = []
    a11_t = []
    c1_t = []
    c2_t = []
    sigma_gamma_11_t = []
    sigma_gamma_22_t = []
    gamma1_t = []
    gamma2_t = []

    # simulate beta-hats
    z1, z2 = simulate()

    # initialize for t0
    a00, a10, a01, a11, c1, c2, gamma1, gamma2, sigma_gamma = initialize()

    # run Gibbs sampling
    for it in range(0, ITS):

        if it % 100 == 0:
            print it

        # draw from conditional distributions
        c1, c2 = draw_c(a00, a10, a01, a11, c1, c2, gamma1, gamma2, z1, z2, debug=False)
        gamma1, gamma2 = draw_gamma(c1, c2, gamma1, gamma2, sigma_gamma, z1, z2, debug=True)
        sigma_gamma = draw_sigma_gamma(gamma1, gamma2, debug=False)
        a00, a10, a01, a11 = draw_a(c1, c2, debug=False)

        # save values from each iteration
        a00_t.append(a00)
        a10_t.append(a10)
        a01_t.append(a01)
        a11_t.append(a11)
        c1_t.append(c1.tolist())
        c2_t.append(c2.tolist())
        sigma_gamma_11_t.append(sigma_gamma[0, 0])
        sigma_gamma_22_t.append(sigma_gamma[1, 1])
        gamma1_t.append(gamma1.tolist())
        gamma2_t.append(gamma2.tolist())

    # evaluate accuracy
    evaluate_a(a00_t, a10_t, a01_t, a11_t)
    evaluate_gamma(gamma1_t, gamma2_t)
    evaluate_sigma_gamma(sigma_gamma_11_t, sigma_gamma_22_t)

if __name__ == "__main__":
    main()
