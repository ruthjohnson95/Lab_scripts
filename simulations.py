from scipy.stats import invwishart
from scipy.special import logit
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    if x < -5:
        x = -5
    return 1 / (1 + math.exp(-x))

random.seed(10)
# set simulation parameters
ITS = 500
BURN = 50

M = 5000
N1 = 200000
N2 = 200000
Ns = 0
H1 = .99
H2 = .99
A00 = .70
A10 = .10
A01 = .10
A11 = 0.10
rho = 0
rho_e = 0
V = np.identity(M)

# gives ~ 2 causals per locus

# generate simulations

x1 = np.random.uniform(0,1,(N1,M))
x1 = (x1 - np.mean(x1, axis=0)) / np.std(x1,axis=0)
x2 = np.random.uniform(0,1, (N2, M))
x2 = (x2 - np.mean(x2, axis=0)) / np.std(x2, axis=0)

c_true = []

s_m_1 = 0
s_m_2 = 0
s_m_12 = 0

for m in range(0,M):
    s_m_1 = s_m_1 + np.dot(x1[:,m], x1[:,m])

sig_11 = ((H1) / ( s_m_1*(A11 + A10) ))
sig_22 = ((H2) / ( s_m_1*(A11 + A01)))

sig_12 = ((rho) / (s_m_1*A11))
sig_21 = sig_12

c = np.random.multinomial(1, [A00, A01, A10, A11], M)

c1_truth = np.zeros(M)
c2_truth = np.zeros(M)

# make true c-vectors
for m in range(0,M):
    if c[m,0] == 1:
        c1_truth[m] = 0
        c2_truth[m] = 0
    elif c[m,1] == 1:
        c1_truth[m] = 1
        c2_truth[m] = 0
    elif c[m,2] == 1:
        c1_truth[m] = 0
        c2_truth[m] = 1
    else:
        c1_truth[m] = 1
        c2_truth[m] = 1

#print np.sum(c1_truth)
#print np.sum(c2_truth)

mu = [0,0]
cov = [[sig_11, sig_12],[sig_21, sig_22]]
sigma_gamma = np.array(cov)
sigma_gamma_truth = sigma_gamma

gamma = np.random.multivariate_normal(mu, cov, M)

gamma1_truth = gamma[:, 0]
gamma2_truth = gamma[:, 1]

beta1 = np.empty(M)
beta2 = np.empty(M)


for m in range(0,M):
    beta1[m] = gamma[m,0] * (c[m,1] + c[m,3])
    beta2[m] = gamma[m,1] * (c[m,2] + c[m,3])

# summary statistics
#Sig_11 = ((1 - H1)) / N1
#Sig_22 = ((1 - H2)) / N2

Sig_11 = ((1 - H1)) / N1
Sig_22 = ((1 - H2)) / N2

Sig_12 = (Ns * rho_e) / (N1 * N2)
Sig_21 = (Ns * rho_e) / (N1 * N2)

mu = np.concatenate( (np.matmul(V, beta1), np.matmul(V, beta2) ))

cov = np.bmat([[Sig_11 * V, Sig_12*V],[Sig_21*V , Sig_22*V]])

z = np.random.multivariate_normal(mu, cov, 1)
z = z.ravel()
z1 = z[0:M]
z2 = z[M:2*M]

####################### end simulations


# make empty list to hold parameters for each iteration
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


# initialize t0: a11, a10, a01, a00
lam = 1
a = np.random.dirichlet([lam, lam, lam, lam], 1)
a = a.ravel()
a00 = a[0]
a10 = a[1]
a01 = a[2]
a11 = a[3]

# TESTING
a00 = A00
a10 = A10
a01 = A01
a11 = A11

print np.sum(c, axis=0)/float(M)

# initialize t0: c1, c2
c1 = np.random.binomial(1, 0.5, M)
c2 = np.random.binomial(1, 0.5, M)


# initialize t0: gammas
gamma1 = np.random.normal(0, .01, M)
gamma2 = np.random.normal(0, .01, M)

# TEST GAMMA: UNCOMMENT BELOW 2 LINES
# true gammas
#gamma1 = gamma[:, 0]
#gamma2 = gamma[:, 1]

# initialize t0: covariance matrix
df0 = M
scale0 = np.identity(2)

sigma_gamma = invwishart.rvs(df0, scale0)

# set variance fixed
sigma_B1 = Sig_11
sigma_B2 = Sig_22

# BEGING: Gibbs Samping

# loop through iterations
for it in range(0, ITS):


    for m in range(0,M):
        # THE BUG IS IN THIS LOOP
        # STEP 1: causal indicator vector
        B1 = (-1/float(2*sigma_B1))*(gamma1[m]*gamma1[m] - 2*z1[m]*gamma1[m]) + logit(a11 + a10)

        b1 = (-1/float(2*sigma_B1))*(gamma1[m]*gamma1[m] - 2*z1[m]*gamma1[m])
        b2 = logit(a11 + a10)
        #B1 = logit(a11 + a10) + ( ( (gamma1[m]*gamma1[m]) - 2*z1[m]*gamma1[m]) /(-2*sigma_B1) )
        dif1 = (gamma1[m]*gamma1[m] - 2*z1[m]*gamma1[m])
        gamma_squared = gamma1[m]*gamma1[m]
        z_gamma = 2*z1[m]*gamma1[m]
        p1 = sigmoid(B1)
        B2 = (-1/float(2*sigma_B2))*(gamma2[m]*gamma2[m] - 2*z2[m]*gamma2[m]) + logit(a11 + a01)

        #B2 = logit(a11 + a01) + ( ( (gamma2[m]*gamma2[m]) - 2*z2[m]*gamma2[m]) /(-2*sigma_B2) )
        dif2 = ( (gamma2[m]*gamma2[m]) - 2*z2[m]*gamma2[m])
        p2 = sigmoid(B2)
        c1m = np.random.binomial(1, p1, 1)
        #if c1m == 1:
            #print "stop"
        c2m = np.random.binomial(1, p2, 1)
        c1[m] = c1m
        c2[m] = c2m

        # STEP 2: effect sizes gammas
        sigma_gamma_11 = sigma_gamma[0, 0]
        sigma_gamma_22 = sigma_gamma[1, 1]

        # params of posterior distribution
        sigma_gamma_pos1 = (sigma_B1 * sigma_gamma_11) / (sigma_B1 + sigma_gamma_11)
        mu_gamma_pos1 = ((z1[m] * c1m) * sigma_gamma_pos1) / sigma_B1
        sigma_gamma_pos2 = (sigma_B2 * sigma_gamma_22) / (sigma_B2 + sigma_gamma_22)
        mu_gamma_pos2 = ((z2[m] * c2m) * sigma_gamma_pos2) / sigma_B2

        # TESTING GAMMAS: COMMENT OUT BELOW 2 LINES
        gamma1[m] = np.random.normal(mu_gamma_pos1, math.sqrt(sigma_gamma_pos1), 1)
        gamma2[m] = np.random.normal(mu_gamma_pos2, math.sqrt(sigma_gamma_pos2), 1)

        #if m == 0:
            #print gamma1_truth[m]
            #print gamma1[m]
            #print '\n'

    # TESTING c-vector:
    #c1[:] = c1_truth[:]
    #c2[:] = c2_truth[:]
    #print np.sum(c1)
    #print np.sum(c2)

    # STEP 3: covariance matrix of gamma
    gamma_two_trait = np.vstack((gamma1, gamma2))
    df = df0 + M
    psi = np.zeros(2)
    for m in range(0, M):
        psi = psi + np.matmul(gamma_two_trait[:, m], np.transpose(gamma_two_trait[:, m]))
    scale = scale0 + psi

    # COMMENTED OUT FOR TESTING:
    sigma_gamma = invwishart.rvs(df, scale)

    # STEP 4: a-probabilities for causal vector

    # TESTING
    #c1[:] = c1_truth[:]
    #c2[:] = c2_truth[:]

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
    print "a00: %f, a10: %f, a01: %f, a11: %f" % (a00, a10, a01, a11)

    # UPDATE: store everything per iteration
    a00_t.append(a00)
    a10_t.append(a10)
    a01_t.append(a01)
    a11_t.append(a11)

    c1_t.append(c1.tolist())
    c2_t.append(c2.tolist())
    sigma_gamma_11_t.append(sigma_gamma[0,0])
    sigma_gamma_22_t.append(sigma_gamma[1,1])
    gamma1_t.append(gamma1)
    gamma2_t.append(gamma2)

# end for-loop through ITS

gamma1_med = np.median(gamma1_t, axis=0)
gamma2_med = np.median(gamma2_t, axis=0)


#print sigma_gamma_truth[0,0]
#print sigma_gamma_truth[1,1]
#print np.median(sigma_gamma_11_t)
#print np.median(sigma_gamma_22_t)

print np.median(a00_t[BURN:])
print np.median(a10_t[BURN:])
print np.median(a01_t[BURN:])
print np.median(a11_t[BURN:])



plt.figure()
plt.plot(range(0,ITS), a00_t)
plt.savefig( 'a00_plot.png' )
plt.figure()
plt.plot(range(0,ITS), a10_t)
plt.savefig( 'a10_plot.png' )
plt.figure()
plt.plot(range(0,ITS), a01_t)
plt.savefig( 'a01_plot.png' )
plt.figure()
plt.plot(range(0,ITS), a11_t)
plt.savefig( 'a11_plot.png' )


plt.figure()
plt.hist(a00_t[BURN:],normed=True)
plt.savefig('a00_hist.png')

plt.figure()
plt.hist(a10_t[BURN:], normed=True)
plt.savefig('a10_hist.png')

plt.figure()
plt.hist(a01_t[BURN:], normed=True)
plt.savefig('a01_hist.png')

plt.figure()
plt.hist(a11_t[BURN:], normed=True)
plt.savefig('a11_hist.png')


