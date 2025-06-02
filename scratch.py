import pandas as pd
import numpy as np
import tensorflow as tf
import time
import munch
import matplotlib.pyplot as plt
#file = pd.read_csv(r'.\Data_files\Brownian_motion.csv')
#print(file.values.shape)
'''
#print(W.shape)
#print(max(W[:,1,4] - file.values[:,212]))
lr_boundaries = np.arange(100, 5000, 100)
print(lr_boundaries, lr_boundaries.shape)
#print(np.linspace(1, 0.1, num = 100, dtype = 'float64'))
lr_values = np.linspace(1e-2, 1e-3, 50)
print(lr_values.shape)
dt = 1.0/52.0
R = np.random.normal(0, np.sqrt(dt), size=(256, 2, 40))
print(R[:,1,:].shape)
#print(np.array([[0.0 for j in range(2)] for i in range(256)]).shape)
#R = tf.concat([np.array([[0.0 for j in range(2)] for i in range(256)]),R], axis = 2)
#print(tf.reduce_sum((a*b)*(a*b), axis=1, keepdims= True))
#print(np.random.uniform(low = 20, high = 30, size = [1]))
t = time.time()
K = 500
#A = np.random.normal(0,1,size = (K,K))
#B = np.random.normal(0,1,size = (1,K))
#C = tf.tensordot(A, B, axes = [[1],[1]])
A = np.array([[1,2,3],[3,4,5]])
B = np.array([[6],[7]])
C = A*B
print(C)
R = np.random.normal(0, np.sqrt(1.0/52.0), size=(256, 200 - 1, 2))
zero_matrix = np.zeros(shape= [256,1, 2])
R = tf.concat([zero_matrix, R], axis = 1)
print(R.shape)
W = tf.cumsum(R, axis=1)
print(W.shape)
time_taken = time.time() - t
print(time_taken)
C = np.zeros([K,K])
print(C)
t = time.time()
for i in range(K):
    for j in range(K):
        C[i,j] = A[i, j]*B[0, j]
print(C)
time_taken = time.time() - t
print(time_taken)

A = np.load('C:/Users/admincshardul/OneDrive/Desktop/Deep_BSDE_solver-master/y0_data.npy')
B = np.load('C:/Users/admincshardul/OneDrive/Desktop/Deep_BSDE_solver-master/loss_data.npy')
print(A,'\n', B)
print(A.shape, B.shape)

dt=1.0/200
num_sample, t_int, W_dim = 10, 50, 2
W = np.random.normal(0, np.sqrt(dt), size=(num_sample, t_int - 1, W_dim))
zero_matrix = np.zeros(shape= [num_sample,1, W_dim], dtype='float64')
W = tf.concat([zero_matrix, W], axis = 1)
#R = np.hstack(([[0.0] for i in range(num_sample)], R))
W = np.cumsum(W, axis=1)  # Full brownian motion with each row being a sample path
print(W[:,0,:], W.shape, W[:,3,:].shape)

A = np.array([[1,2,3],[3,4,5]])
B = np.array([[1,0,1],[0,1,0]])
print(A*B)
print(tf.reduce_sum(A*B, axis = 1, keepdims= True))

print(tf.config.list_physical_devices('GPU'))

A = {"a": 3, "b": 4}
A = munch.munchify(A)
print(A)
#print(A["b"])'''
'''A = [1, 2, 3]
B = [1, 2, 3]
C = list(zip(A,B))
np.random.seed(14)
np.random.shuffle(A)
np.random.seed(14)
np.random.shuffle(B)
print(A,B)
print(C)
tf.keras.backend.set_floatx('float32')
dt = 1.0/52.0
W = np.random.normal(0, np.sqrt(dt), size=(64, 52 - 1, 1))
zero_matrix = np.zeros(shape=[64, 1, 1], dtype='float32')
W = np.concatenate([zero_matrix, W], axis=1)
W = np.cumsum(W, axis=1)                        # Full brownian motion as a tensor
T_grid = dt*np.array([[[t] for t in range(52)]
                     for _ in range(64)])
S = 72.139*np.exp((365.0*0.0011 - 365.0*(0.0162**2)/2)*T_grid + np.sqrt(365.0)*0.0162*W)
S = S/(600.0*np.exp(0.026*T_grid) + 5.54478*S)

plt.figure()
plt.plot(T_grid[0], S[0,:,0],label="S")
plt.plot(T_grid[1], S[1,:,0],label="S")
plt.show()'''

'''a = np.array([[1, 2], [3, 4], [5, 6]])
b = a
print(tf.stack((a, b), axis=1))
print(1 + a)
print(tf.reduce_sum(a, axis=0))
print(tf.reduce_sum(a, axis=0, keepdims=True))
print(tf.pow(np.e, a))'''

'''all_ones_n = tf.ones(shape=(3, 3, 3), dtype="float64")
a = np.random.uniform(low=0.1, high=0.8, size=[3])
print(tf.reduce_sum(a[:, None]*tf.ones(shape=(3))[None, :], axis=1, keepdims=True))

print(a[None, None, :], np.shape(a[None, None, :]), '\n', all_ones_n*a, '\n', all_ones_n*a[:, None, None])
print(tf.reduce_sum(all_ones_n*a[:, None, None], axis=-1, keepdims=False))'''
#print(a,'\n', a**2)
#print(tf.reduce_sum(a**2, axis=1, keepdims=True))
#print(all_ones_n*a)
''' SAMPLE FUCN
sig = np.sqrt(365.0)*np.array([[0.016239783125394393, 0.0], [0.00474972518872159, 0.011421878446647651]])
b = 365.0*np.array([0.0010924644457478037, 0.0002029013694769593])
S_init = np.array([72.13994598388672, 113.17442321777344])
gamma = np.array([400.0, 4.1586, 2.651])
r = 0.026
dt = 1.0/52.0
W_dim = 2
num_sample = 4
t_grid_size = 5

T_grid = dt*np.array([[t for t in range(t_grid_size)] for _ in range(num_sample)])
W = np.random.normal(0, np.sqrt(dt), size=(num_sample, t_grid_size - 1, W_dim))
zero_matrix = np.zeros(shape=[num_sample, 1, W_dim], dtype='float64')
W = np.concatenate([zero_matrix, W], axis=1)
W = np.cumsum(W, axis=1)
print("shapes_W_T", np.shape(W), np.shape(T_grid))

S = [S_init[i]*np.exp((b[i] - tf.reduce_sum(sig[i]**2)/2)*T_grid +
            tf.reduce_sum(sig[i][None, None, :]*W, axis=2, keepdims=False))
            for i in range(W_dim)]
S = [S[i]/(gamma[0]*np.exp(r*T_grid) + tf.reduce_sum(gamma[1:, None, None]*S, axis=0)) for i in range(W_dim)]
print(np.shape(S), S)'''

a = np.random.uniform(low=0.1, high=0.8, size=[3, 2])
b = np.random.uniform(low=0.1, high=0.8, size=[3, 1])
print(a, b, a*b)
print(np.append(np.array([[1], [2]]), [[0.6], [3]], axis=-1))

A = np.array([[1, 2, 3], [4, 6, 8]])
B = np.array([[1, 2, 3], [2, 3, 4]])
print(A/B)