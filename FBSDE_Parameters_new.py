'''This file defines the FBSDE class for solving Forward-Backward Stochastic Differential Equations (FBSDEs)
   with specific parameters from our portfolio optimization problem. It uses the config file "FBSDE_config_1d.json",
    market parameters from "Real_data_GBM.py" and "saved_processes_simdata.py" and the target distribution from
    "targ_dist.py" for setting up the FBSDE parameters.'''

import numpy as np
import tensorflow as tf
from targ_dist import TargDist
import json
import munch

conf_path = "configs\FBSDE_config_1d.json"
stock_data = r'Data_files\b_and_sig.json'
with open(conf_path) as json_config, open(stock_data) as json_stock_dat:
    config = json.load(json_config)
    b_sig_dat = json.load(json_stock_dat)

config = munch.munchify(config)
b_sig_dat = munch.munchify(b_sig_dat)

class Base(object):
    '''Base class for defining FBSDEs'''

    def __init__(self, eqn_config):
        self.X_dim = eqn_config.X_dim
        self.Y_dim = eqn_config.Y_dim
        self.W_dim = eqn_config.W_dim
        self.Z_shape = [self.Y_dim, self.W_dim]
        self.total_time = eqn_config.total_time
        self.t_grid_size = eqn_config.t_grid_size
        self.delta_t = 1/self.t_grid_size
        self.y_init = None


    def f(self, t, x, y, z):
        raise NotImplementedError

    def g(self, t, x):
        raise NotImplementedError


class FBSDE(Base):
    ''' Class for setting parameters for the portfolio optimization propblem with trading speed '''

    def __init__(self, eqn_config, b_sig_dat, alpha):
        super(FBSDE, self).__init__(eqn_config)
        self.X_init = eqn_config.X_init
        self.eqn_config = eqn_config
        self.r = eqn_config.r
        self.rho = eqn_config.rho
        self.alpha = alpha
        self.k = np.linspace(0.95, 1.05, 11)

        # Conversion of drift and volatility from daily to yearly values
        # b_sig_dat is a dictionary with keys 'drift', 'volatility' and 'S_init'
        self.gamma = np.array(eqn_config.gamma)
        self.b = 365.0*np.array(b_sig_dat.drift)
        self.sig = np.sqrt(365.0)*np.array(b_sig_dat.volatility)
        self.S_init = np.array(b_sig_dat.S_init)

        self.lamb = self.eqn_config.lamb       # value of the smoothing parameter

    def aV(self, s):            #shape: (n_sample, 1)
        #1d case: return tf.pow((s*self.gamma[1]*self.sig[0]), 2) - s*self.gamma[1]*(self.b[0] - self.r)
        #t1 = tf.reduce_sum(tf.pow(sum([(s[i, :, None]*self.gamma[i+1]*self.sig[i, None, :]) for i in range(self.eqn_config.W_dim)]), 2),
        #                     axis=1, keepdims=True)
        #t2 = sum([(s[i, :, None]*self.gamma[i+1]*(self.b[i] - self.r)) for i in range(self.eqn_config.W_dim)])
        #return t1 - t2
        t3 = tf.pow((s * self.gamma[1] * self.sig[0, 0]), 2) - s * self.gamma[1] * (self.b[0] - self.r)
        #print("aV", t3.shape)
        return t3

    def ad(self, s):            #shape: (n_stock, sample, 1)
        #1d case: return s*(self.b[0] - self.r) - tf.pow((s*self.sig[0]),2)*self.gamma[1]
        #t1 = tf.stack([(s[i, :, None]*(self.b[i] - self.r)) for i in range(self.eqn_config.W_dim)])
        #print("t1", np.shape(t1))
        #t2 = sum([(s[k, :, None]*self.gamma[k+1]*self.sig[k, None, :]) for k in range(self.eqn_config.W_dim)])
        #print("t2", np.shape(t2))
        #t3 = [s[i, :, None]*self.sig[i, None, :] for i in range(self.eqn_config.W_dim)]
        #print("t3", np.shape(t3))
        #t4 = tf.stack([tf.reduce_sum(t2*t3[i], axis=1, keepdims=True) for i in range(self.eqn_config.W_dim)])
        #print("t4", np.shape(t4))
        #return t1 - t4
        t5 = s * (self.b[0] - self.r) - tf.pow((s * self.sig[0, 0]), 2) * self.gamma[1]
        #print("ad", t5.shape)
        return t5

    def bV(self, s):            #shape: (sample, W_dim)
        #1d case: return -s*self.gamma[1]*self.sig[0]
        # change sig[i, None, 0] for multidimensional case
        #t1 = -sum([(s[i, :, None]*self.gamma[i+1]*self.sig[i, None, 0]) for i in range(self.eqn_config.W_dim)])
        #print(np.shape([(s[i, :, None]*self.gamma[i+1]*self.sig[i, None, :]) for i in range(self.eqn_config.W_dim)]))
        #return t1
        t2 = -s * self.gamma[1] * self.sig[0, 0]
        #print("bV", t2.shape)
        return t2

    def bd(self, s):            #shape: (stock, sample, W_dim)
        #1d case  return s*self.sig[0]
        # change sig[i, None, 0] for multidimensional case
        #t1 = tf.stack([s[i, :, None]*self.sig[i, None, 0] for i in range(self.eqn_config.W_dim)])
        #return t1
        t2 = s * self.sig[0, 0]
        #print("bd", t2.shape)
        return t2

    def f(self, t, x, y, z, s, zeta):
        # 1-d case in comments
        #print("test 1", x[0][0], Exp_x)
        #f0 = y[0]*self.aV(s) + tf.reduce_sum(z[0]*self.bV(s), axis=1, keepdims=True) - self.rho*y[0] + 1.0*self.ldx(t, x[0], Exp_x)
        #print('t1', np.shape(y[0]*self.aV(s)), np.shape(tf.reduce_sum(z[0]*self.bV(s), axis=1, keepdims=True)),
              #np.shape(self.rho*y[0]))

        #f0 = y[0]*self.aV(s) + z[0]*self.bV(s) - self.rho*y[0] + 1.0 * self.ldx(t, x[0], Exp_x)
        #if t >= 49:
        #    print("test_shapes:", y[0].shape, x[0].shape)
        #    print("test values:", y[0], x[0])
        lamb_til = 0.0
        f0 = y[0] * self.aV(s) + z[0] * self.bV(s) - self.rho * y[0] \
             + self.alpha * self.ldx(t, x[0], zeta) \
             + lamb_til*(-2*x[0]/(1+(x[0])**2)**2)

        #f1 = [y[0]*self.ad(s)[i] + tf.reduce_sum(z[0]*self.bd(s)[i], axis=1, keepdims=True) - self.rho*y[1][i]
        #      for i in range(self.eqn_config.W_dim)]

        #f1 = y[0]*self.ad(s) + z[0]*self.bd(s) - self.rho*y[1]
        f1 = y[0] * self.ad(s) + z[0] * self.bd(s) - self.rho * y[1]

        #print('f1', np.shape(f1))
        return f0, f1

    def g(self, t, x):
        return 0.0

    def ldx(self, t, x0, zeta):
        # Takes the stock and brownian motion data and returns the array for the L-derivative at every time step
        # and sample point.
        #nu_supp = [0.96, 0.98, 1.0, 1.02, 1.04]
        pi = TargDist()
        k = self.k #self.eqn_config.nu_support             # Support for target distribution
        #k = np.linspace(0.95, 1.05, 7)
        #k = np.array([0.96, 0.98, 1.0, 1.02, 1.04])
        #Exp_x0 = tf.reduce_mean(x0, axis=-1)
        #print(Exp_x0)
        #T_grid = self.delta_t * np.array([t for t in range(self.eqn_config.t_grid_size)])
        # Compute at the beginning
        p = [pi.p(k=k[i], T=(t/(self.eqn_config.total_time*self.eqn_config.t_grid_size))) for i in range(len(k))]

        ldx = 0.0
        for i in range(len(k)):
            #ldx += (1/len(k))*(tf.cast(-1/(1 + tf.pow(np.e, (-self.lamb*(k[i] - Exp_x0 - p[i])))), tf.float64)) * \
            #       tf.cast(1/(1 + tf.pow(tf.cast(np.e, tf.float64), -self.lamb*(k[i] - x0))), tf.float64)
            ldx += (1 / len(k)) * (tf.cast(-1 / (1 + tf.exp((self.lamb*p[i] - tf.reduce_mean(zeta[i])))), tf.float32)) * \
                   tf.cast(1/(1 + tf.pow(tf.cast(np.e, tf.float32), -self.lamb*(k[i] - x0))), tf.float32)

        #print("ldx", ldx.shape, ldx[0])
        return ldx
        #return -tf.cast(2.0, tf.float64) * tf.maximum(tf.zeros_like(x0), tf.cast(k[0], tf.float64) - x0)


    def dist(self, t, x0):
        pi = TargDist()
        k = self.k #self.eqn_config.nu_support
        p = [pi.p(k=k[i], T=(t / (self.eqn_config.total_time*self.eqn_config.t_grid_size))) for i in range(len(k))]

        temp = []
        for i in range(len(k)):
            temp.append(np.mean((1/self.lamb)*np.log(1 + np.e**(self.lamb*(k[i]-x0))), axis=0))

        temp2 = 0
        for i in range(len(k)):
            temp2 += (1/len(k))*(1/self.lamb)*np.log(1 + np.e**(self.lamb*(temp[i] - p[i])))

        return temp2

    def L(self, data, t):
        # Change of measure
        dW = data[0][:, 1:] - data[0][:, :-1]
        S = data[1]
        return tf.exp(tf.reduce_sum((self.bV(S)*dW - 0.5*tf.square(self.bV(S))*self.delta_t)[:, :t],
                                    axis=1, keepdims=False))


class FBSDE_verification(Base):
    ''' Example 2 from the paper "Convergence of Deep BSDE method for Coupled FBSDE" by Han and Long (2022) '''

    def __init__(self, eqn_config):
        super(FBSDE_verification, self).__init__(eqn_config)
        self.X_init = np.array([np.pi for _ in range(self.X_dim)])
        self.ExplicitSolution = np.exp(-0.1*self.total_time)*0.1*self.W_dim
        self.sig = 0.3
        self.r = 0.1
        self.D = 0.1

    def f(self, t, x, y, z):
        return -self.r*y + 0.5*(self.sig**2)*np.exp(-3*self.r*(1-(t*self.delta_t))) * \
               tf.pow(self.D*tf.reduce_sum(tf.sin(x), axis=1, keepdims=True), 3)

    def g(self, t, x):
        return self.D*tf.reduce_sum(tf.sin(x), axis=1)