# add t_int, W_dim, y_init_range matrix, delta_t, in config
# x_0 needs to be column vector of samples
import numpy as np
import tensorflow as tf
import logging
import pandas as pd
import time

DELTA_CLIP = 50.0

class FBSDEsolver(object):
    def __init__(self, config):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.model = full_NN(config)
        self.y_init = self.model.y_init
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule, epsilon=1e-8)
        self.W_file = pd.read_csv('Data_files\Brownian_motion.csv')

    def Sample(self, num_sample):
        # Random list to sample mini-batches.
        batch_index = np.random.randint(0, self.net_config.valid_size, num_sample)
        W = np.zeros(shape = [num_sample,self.eqn_config.W_dim,self.net_config.t_int], dtype='float64')
        for i in range(num_sample):
            for d in range(self.eqn_config.W_dim):
                for t in range(self.net_config.t_int):
                    W[i,d,t] = self.W_file.values[batch_index[i],d*self.net_config.t_int + t]
        '''dt = self.eqn_config.num_time_interval / self.eqn_config.t_grid_size
        R = np.random.normal(0, np.sqrt(dt), size=(num_sample, self.eqn_config.W_dim, self.net_config.t_int - 1))
        #R = tf.concat([np.array([[0.0 for j in range(self.eqn_config.W_dim)] for i in range(num_sample)]), R], axis = 2)
        #R = np.hstack(([[0.0 for j in range(self.eqn_config.W_dim)] for i in range(num_sample)] , R))
        for d in range(self.eqn_config.W_dim):
            R[:,d,:] = np.hstack([[0.0] for i in range(num_sample)], R[:,d,:])
        W = np.cumsum(R, axis=2)  # Full brownian motion with each row being a sample path'''

        return W

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.Sample(self.net_config.valid_size)

        # begin sgd iteration
        for step in range(self.net_config.num_iterations + 1):
            #batch_index = np.random.randint(0,self.net_config.valid_size,self.net_config.batch_size)
            if step % self.net_config.logging_frequency == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                y_init = self.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config.verbose:
                    logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                        step, loss, y_init, elapsed_time))
            self.train_step(self.Sample(self.net_config.batch_size))
        return np.array(training_history)

    def loss_fn(self, input, training):
        #W, x = input
        y_terminal = self.model(input, training)
        delta = tf.abs(y_terminal)
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(delta < DELTA_CLIP, tf.square(delta),
                                       2 * DELTA_CLIP * delta - DELTA_CLIP ** 2))

        return loss

    def grad(self, input, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(input, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training = True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))


class full_NN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.t_int = config.net_config.t_int
        self.y_init = tf.Variable(np.random.uniform(low = self.net_config.y_init_range[0],
                                                    high = self.net_config.y_init_range[1],
                                                    size = [1]))
        self.z_init = tf.Variable(np.random.uniform(low = -0.1, high = 0.1,
                                                    size = [1, self.eqn_config.W_dim]))
        self.subnet = [Subnet(config) for _ in range(self.t_int - 1)]

    def call(self, input, training):
        W = input
        delta_t = self.eqn_config.num_time_interval/self.eqn_config.t_grid_size
        a1, b1 = 2.0, 3.0
        a2, b2 = np.array([[1.0,2.0]]), np.array([[2.0,3.0]])
        mu = -0.3
        Lz = 0.4
        c = 0.2
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(W)[0], 1]), dtype="float64")
        x = all_one_vec
        y = all_one_vec * self.y_init
        #print(y)
        #print(y.shape)
        z = tf.matmul(all_one_vec, self.z_init)
        #print(z.shape)
        #print(tf.reduce_sum(z,1).shape)
        #time_grid = np.arange(0, self.net_config.t_int)*self.net_config.delta_t
        for t in range(self.t_int - 1):
            x = x+ (a1*x + b1*y)*delta_t + tf.reduce_sum((W[:,:,t+1]-W[:,:,t])*(tf.matmul(x,a2)
                + tf.matmul(y,b2)), axis = 1, keepdims = True)
            #print(t, x.shape)
            y = y - (mu*y + Lz*tf.reduce_sum(z, 1, keepdims = True) + c)*delta_t\
                + tf.reduce_sum(z*(W[:,:,t+1]-W[:,:,t]), 1, keepdims=True)
            #print(t,y.shape)
            z = self.subnet[t](x, y, training) /self.eqn_config.W_dim
            #print(t,z.shape)
        # Terminal time Y_T
        y = y - (mu * y + Lz * tf.reduce_sum(z, 1, keepdims = True) + c) * delta_t \
            + tf.reduce_sum(z * (W[:, :, -1] - W[:, :, -2]), 1, keepdims=True)

        return y


class Subnet(tf.keras.Model):
    def __init__(self,config):
        super().__init__()
        W_dim = config.eqn_config.W_dim
        num_hiddens = [120, 120]
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum = 0.99,
                epsilon = 1e-6,
                beta_initializer =  tf.random_normal_initializer(0.0, stddev = 0.1),
                gamma_initializer = tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens)+2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias = False,
                                                   activation = None)
                             for i in range(len(num_hiddens))
                             ]
        self.dense_layers.append(tf.keras.layers.Dense(W_dim, activation = None))

    def call(self, x, y, training):
        x = tf.concat([x,y],1)
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers)-1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        #print(x.shape)
        x = self.bn_layers[-1](x, training)
        #print(x.shape)
        return x
