'''
This program simulates the GBM corresponding to the parameter estimates from "Real_data_GBM.py" and computes the
relevant process required for the FBSDE and saves them in a DataFrame format in a '.csv' file along with the
simulated Brownian motion.
'''

import numpy as np
import ast
import json
import munch
import datatable as dt
import matplotlib.pyplot as plt
import pandas as pd

# Reading daily drift and volatility data from "b_and_sig.dat"

file = open("./Data_files/b_and_sig.dat", 'r')
data = file.readlines()
#print(data)
dict1 = ast.literal_eval(data[0])
#print(dict1)
b = dict1['drift']
sig = dict1['volatility']
file.close()

# Reading the config file for time interval and sample size parameters.

config = json.load(open("configs\FBSDE_config.json"))
config = munch.munchify(config)
T = config.eqn_config.t_grid_size                   # We normalize the time interval to 1 and have 'T' weeks
dt = config.eqn_config.num_time_interval/T


# Convert b and sigma from daily drift and volatility to yearly (when num_time_interval=1):

b = T*7*np.array(b)
sig = np.sqrt(T*7)*np.array(sig)
print(b,sig)

# Initialisation of parameters and Brownian motion:

T_grid = [dt*i for i in range(T)]
D = config.net_config.valid_size              # number of sample paths, D = 256; 2-dim Brownian motion

R = np.random.normal(0, np.sqrt(dt), size = (D, T-1))                   # D rows and T columns of std. normal
R = np.hstack([[0.0] for a in range(D)], R)
W1 = np.cumsum(R, axis=1)                   # Full brownian motion with each row being a sample path
R = np.random.randn(D, T-1)                   # Doing the same thing for second Brownian motion
R = np.hstack([[0.0] for a in range(D)], R)
W2 = np.cumsum(R, axis=1)
dict1 = {'W1_%d'%i : W1[:,i] for i in range(T)}
dict2 = {'W2_%d'%i : W2[:,i] for i in range(T)}
dict1.update(dict2)
df = pd.DataFrame(dict1)
#print(df.head())
df.to_csv(r'Data_files/Brownian_motion.csv', index = False)
#df1 = pd.read_csv('Data_files\Brownian_motion.csv')
#print(df1.head())

# Stock price and associated processes simulation: (k->time; j->sample path; i->asset)

r = 0.026                                   # weekly interest rate, r = 0.05% per week or 2.6% p.a.
#b = np.array([0.003, 0.004])                # weekly drift
#sig = np.array([[0.03, 0.0], [0.0, 0.05]])  # weekly volatility; uncorrelated stocks at this moment
tS = np.array([[[0.0 for k in range(T)] for a in range(D)] for i in range(3)])
tS0 = np.append(1.0, ast.literal_eval(data[1]))
#print(tS0)
#tS0 = np.array([1.0, 10.0, 20.0])                 # 'tS' denotes the original stock prices and 'S' denotes relative stock prices
S = np.array([[[0.0 for k in range(T)] for a in range(D)] for i in range(3)])
gamma = np.array([200.0, 40.0, 20.0])             # Naive portfolio initialisation; Initial wealth = 1000 euros.

for a in range(D):                          # Risk-free asset
    for k in range(T):
        tS[0, a, k] = tS0[0]*np.exp(r*k*dt)

for i in range(2):                          # Risky assets
    for a in range(D):
        for k in range(T):
            tS[i+1, a, k] = tS0[i+1]*np.exp((b[i] - 0.5*(np.sum(sig[i, :]))**2)*k*dt + (sig[i, 0]*W1[a,k] + sig[i, 1]*W2[a,k]))

for i in range(3):                          # Relative asset prices
    for a in range(D):
        for k in range(T):
            S[i, a, k] = tS[i, a, k]/np.sum(gamma[:]*tS[:, a, k])

# Defining the parameters of the relative wealth process:

aV = np.array([[0.0 for t in T_grid] for a in range(D)])             
ad = np.array([[[0.0 for t in T_grid] for a in range(D)] for i in range(2)])
bV = np.array([[[0.0 for t in T_grid] for a in range(D)] for j in range(2)])
bd = np.array([[[[0.0 for t in T_grid] for a in range(D)] for j in range(2)] for i in range(2)])

for a in range(D):                                                      #\alpha^V_t
    for k in range(T):
        aV[a,k] = np.sum(np.matmul(S[1:, a, k]*gamma[1:],sig[:,:])**2)
dict1 = {'aV_%d'%k: aV[:,k] for k in range(T)}
df = pd.DataFrame(dict1)
df.to_csv(r'Data_files/aV.csv', index=False)
#df = pd.read_csv(r'Data_files\aV.csv')
#print(df)

for i in range(2):                                                      # \alpha^\delta_t
    for a in range(D):
        for k in range(T):
            ad[i,a,k] = S[i+1,a,k]*(b[i]-r) - np.sum(np.matmul(S[1:, a, k]*gamma[1:],sig[:,:])*(S[i+1,a,k]*sig[i,:]))
dict1 = {}
for i in range(2):
    dict1.update({('ad^{}_{})'.format(i, k)) : ad[i,:,k] for k in range(T)})
#print(dict1)
df = pd.DataFrame(dict1)
df.to_csv(r'Data_files/ad.csv', index=False)
#df = pd.read_csv(r'Data_files\ad.csv')
#print(df)

for j in range(2):                          # \beta^V_t
    for a in range(D):
        for k in range(T):
                bV[j,a,k] = -np.sum(np.matmul(S[1:,a,k]*gamma[1:],sig[:,j]))
dict1 = {}
for j in range(2):
    dict1.update({('bV^{}_{}'.format(j, k)): bV[j,:,k] for k in range(T)})
df = pd.DataFrame(dict1)
df.to_csv(r'Data_files/bV.csv', index=False)
#df = pd.read_csv(r'bV.csv')
#print(df.values.shape)

for i in range(2):                          # \beta^\delta_t
    for j in range(2):
        for a in range(D):
            for k in range(T):
                bd[i,j,a,k] = S[i+1,a,k]*sig[i,j]
dict1 = {}
for i in range(2):
    for j in range(2):
        dict1.update({('bd^({},{})_{}'.format(i,j,k)): bd[i,j,:,k] for k in range(T)})
df = pd.DataFrame(dict1)
df.to_csv(r'Data_files/bd.csv', index=False)
#df = pd.read_csv(r'bd.csv')
#print(df)
#print(df['bd^(1,0)_3'].values)


#pictle.dump()
#plt.figure()
#plt.plot(T_grid, W1[0],label="aV")
#plt.plot(T_grid, bd[0,0,0],label="ad")
#plt.show()


