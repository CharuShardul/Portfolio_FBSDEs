import numpy as np
from scipy.stats import norm as norm
import matplotlib.pyplot as plt
import pandas as pd


class TargDist:
    '''Class for storing the target distribution and the parameter p_k. The target distribution pi_t is taken as a GBM
    with drift 'mu' and volatility 'sigma'. "p_k" is calculated using  Black-Scholes formula for European put option.'''
    def __init__(self, mu=0.05, sigma=0.05):
        self.mu = mu
        self.sigma = sigma

    def _d1(self, k, T):

        return (1/(self.sigma*np.sqrt(T+1)))*(np.log(1/(k*np.e**(-self.mu*(T+1)))) + (T+1)*(self.sigma**2)/2)

    def _d2(self, k, T):
        return self._d1(k, T) - self.sigma*np.sqrt(T)

    def _N(self, x, loc=0.0, scale=1.0):
        # returns cdf of standard normal distribution by default
        return norm.cdf(x, loc, scale)

    def p(self, k, T):
        return np.e**(self.mu*T)*(k*np.e**(-self.mu*T)*self._N(-self._d2(k, T)) - self._N(-self._d1(k, T)))

    def pdf(self, x, T):
        # returns the pdf of the target distribution at time t
        return norm.pdf(x, loc=self.mu*T, scale=self.sigma*np.sqrt(T))


pi = TargDist()
T = 0.5
k_supp = [0.96, 0.98, 1.0, 1.02, 1.04]
#k_supp = [1.0]
print([pi.p(k_supp[i] ,T) for i in range(len(k_supp))])

'''
r = mx.rvs(size = 10000, scale=np.sqrt(1/2))
fig, ax = plt.subplots(1,1)
ax.hist(r, 100, density = True)
plt.show()
k_seq = np.linspace(0.0, 0.9, 10)
quantiles = np.quantile(r,k_seq)
print(quantiles)
#print(np.maximum(r- 0.5,r).size)

# Calculation of p_k:

p = np.zeros(len(k_seq))
for k in range(len(k_seq)):
    rv = np.maximum(quantiles[k] - r, np.zeros(10000))
    p[k] = np.mean(rv)
print(p)
#print(list(zip(quantiles,p)))
df = pd.DataFrame(list(zip(quantiles, p)), columns=['k', 'p_k'])
print(df)
df.to_csv(r'Data_files/targ_parameters.csv', index=False)
'''