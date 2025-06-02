
import numpy as np
import tensorflow as tf
import logging
import time

DELTA_CLIP = 50.0


class FBSDEsolver(object):
    def __init__(self, config, equation):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.equation = equation
        self.model = fullNN(equation, config)
        self.b = equation.b
        self.sig = equation.sig
        self.S_init = equation.S_init
        self.y_init = [self.model.y_init_0, self.model.y_init_1]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)
        self.fic_num = self.eqn_config.fict_play_num

    def Sample(self, num_sample):
        # Generating mini-batch samples for Brownian motion and relative stock price process
        #self.b = 0
        dt = self.equation.delta_t
        W = np.random.normal(0, np.sqrt(dt), size=(num_sample, self.eqn_config.t_grid_size - 1, self.eqn_config.W_dim))
        zero_matrix = np.zeros(shape=[num_sample, 1, self.eqn_config.W_dim], dtype='float64')
        W = np.concatenate([zero_matrix, W], axis=1)
        W = np.cumsum(W, axis=1)                        # Full brownian motion as a tensor
        T_grid = dt*np.array([[[t] for t in range(self.eqn_config.t_grid_size)]
                             for _ in range(num_sample)])
        S = self.S_init*np.exp((365.0*self.b - 365.0*(self.sig**2)/2)*T_grid + np.sqrt(365.0)*self.sig*W)
        S = S/(self.eqn_config.gamma[0]*np.exp(self.eqn_config.r*T_grid) + self.eqn_config.gamma[1]*S)

        return W, S

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.Sample(self.net_config.valid_size)

        for p in range(self.fic_num):
            # begin sgd iteration
            for step in range(self.net_config.num_iterations + 1):
                if step % self.net_config.logging_frequency == 0:
                    #print(len(self.model.trainable_variables))
                    #print(self.model.summary)
                    loss = self.loss_fn(valid_data, training=False).numpy()
                    y_init = [self.y_init[0].numpy()[0], self.y_init[1].numpy()[0]]
                    elapsed_time = time.time() - start_time
                    #print(self.eqn_config.explicit_solution)
                    '''if self.eqn_config.explicit_solution == True:
                        rel_err = (np.exp(-0.1 * self.eqn_config.total_time) * 0.1 * self.eqn_config.W_dim - y_init) \
                                / (np.exp(-0.1 * self.eqn_config.total_time) * 0.1 * self.eqn_config.W_dim)
                        training_history.append([step, loss, y_init, rel_err, elapsed_time])
                    else:'''
                    training_history.append([p, step, loss, y_init[0], y_init[1], elapsed_time])
                    '''if self.net_config.verbose and self.eqn_config.explicit_solution == True:
                        #rel_err = (np.exp(-0.1 * self.eqn_config.total_time) * 0.1 * self.eqn_config.W_dim - y_init) \
                                  #/ (np.exp(-0.1 * self.eqn_config.total_time) * 0.1 * self.eqn_config.W_dim)
                        logging.info("step: %5u,    loss: %.4e,  Y0: %.4e,  relative error: %.4e, elapsed time: %3u" % (
                            step, loss, y_init, rel_err, elapsed_time))
                    elif self.net_config.verbose == True:'''
                    logging.info("fict_num: %3u,   step: %5u,   loss: %.4e,  Y0[0]: %.4e, Y0[1]: %.4e,  elapsed time: %3u"
                                 % (p, step, loss, y_init[0], y_init[1], elapsed_time))

                self.train_step(self.Sample(self.net_config.batch_size))

        return np.array(training_history)

    def loss_fn(self, input, training):
        #x_terminal, y_terminal = self.model(input, training)
        x0, x1, y0, y1 = self.model(input, training)
        delta = tf.square(tf.abs(y0)) + tf.square(tf.abs(y1))
        loss = tf.sqrt(tf.reduce_mean(delta))
        #loss = tf.reduce_mean(tf.square(delta))

        return loss

    def grad(self, input, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(input, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))


class fullNN(tf.keras.Model):
    def __init__(self, equation, config):
        super(fullNN, self).__init__()
        self.equation = equation
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.t_grid_size = config.eqn_config.t_grid_size
        self.y_init_0 = tf.Variable(np.random.uniform(low=self.net_config.y_init_range[0],
                                                    high=self.net_config.y_init_range[1],
                                                    size=[1]))
        self.y_init_1 = tf.Variable(np.random.uniform(low=0.5,
                                                      high=0.8,
                                                      size=[1]))
        self.z_init_0 = tf.Variable(np.random.uniform(low=-0.1, high=0.1, size=[1]))
        self.z_init_1 = tf.Variable(np.random.uniform(low=-0.1, high=0.1, size=[1]))
        self.subnet = [Subnet(config) for _ in range(self.t_grid_size - 1)]

    def call(self, input, training):
        W, S = input
        dt = self.eqn_config.total_time/self.eqn_config.t_grid_size
        x = self.equation.X_init#np.array([1.0, 0.0])
        #all_one_vec = tf.ones(shape=tf.stack([tf.shape(W)[0], 1]), dtype="float64")
        all_ones = tf.ones(shape=tf.stack([tf.shape(W)[0], 1]), dtype="float64")
        #all_zeros = tf.zeros(shape=[tf.shape(W)[0], 1], dtype="float64")
        #print(tf.shape(all_zeros))
        #all_one_mat_2 = tf.Variable(tf.ones(shape=([tf.shape(W)[0, self.equation.X_dim]]), dtype="float64"))
        #all_one_mat_3 = tf.Variable(tf.ones(shape=tf.stack([tf.shape(W)[0, self.equation.X_dim]]), dtype="float64"))
        x0 = all_ones * 1.0#* x[0]   #np.array(self.equation.X_init)
        x1 = all_ones * 0.0#* x[1]
        x = tf.stack((x0,x1))
        #print("x:", tf.shape(x))
        y0 = all_ones * self.y_init_0
        y1 = all_ones * self.y_init_1
        #y = tf.stack((y0, y1), axis=0)
        y = (y0, y1)
        z0 = all_ones * self.z_init_0
        z1 = all_ones * self.z_init_1
        #z = tf.stack((z0, z1),axis=0)
        z = (z0, z1)
        #print("y:", tf.shape(y), "z:", tf.shape(z))
        #print("S shape =", tf.shape(tf.concat([S[:, 0], S[:, 0]], axis=1)))
        for t in range(self.eqn_config.t_grid_size -1):                         # optimize aV, bV, etc. calls.
            x0 = x0 + (self.equation.aV(S[:, t])*x0 + self.equation.ad(S[:, t])*x1)*dt + \
                 (self.equation.bV(S[:, t]) * x0 + self.equation.bd(S[:, t]) * x1)*(W[:, t+1]-W[:, t])
            #if t == 0:
                #print("a = ", tf.shape(self.equation.f(t, x, y, z, S[:, t])[0]))
                #print("b = ", tf.shape(z0))
            #x = tf.stack((x0, x1), axis=1)
            x = (x0, x1)
            y0 = y0 - self.equation.f(t, x, y, z, S[:, t])[0]*dt + z0*(W[:, t+1]-W[:, t])
            y1 = y1 - self.equation.f(t, x, y, z, S[:, t])[1]*dt + z1*(W[:, t+1]-W[:, t])
            x1 = x1 - y1*dt
            z = tf.concat((x0, x1, y0, y1), axis=1)
            #print("z shape", type(z), tf.shape(z))
            z = self.subnet[t](z, training)
            y = (y0, y1)
            #z = (z0, z1)
            #z0, z1 = z[:, 0, np.newaxis], z[:, 1, np.newaxis]
            z0, z1 = tf.keras.layers.Lambda(lambda z: z[0, np.newaxis])(z), \
                        tf.keras.layers.Lambda(lambda z: z[1, np.newaxis])(z)
        x0 = x0 + (self.equation.aV(S[:, -1]) * x0 + self.equation.ad(S[:, -1]) * x1) * dt + \
             (self.equation.bV(S[:, -1]) * x0 + self.equation.bd(S[:, -1]) * x1) * (W[:, -1] - W[:, -2])
        y0 = y0 - self.equation.f(t, x, y, z, S[:, -2])[0] * dt + z0 * (W[:, -1] - W[:, -2])
        y1 = y1 - self.equation.f(t, x, y, z, S[:, -2])[1]*dt + z1*(W[:, -2]-W[:, -1])
        x1 = x1 - y1*dt

        return x0, x1, y0, y1


class Subnet(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        Z_dim = config.eqn_config.X_dim
        num_hiddens = [10 + config.eqn_config.X_dim, 10 + config.eqn_config.X_dim]
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                axis=1,
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens)+2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))
                             ]
        self.dense_layers.append(tf.keras.layers.Dense(Z_dim, activation=None))

    def call(self, z, training):
        #z = tf.concat([x, y], axis=1)
        #z = tf.concat([x0, y0], axis=1)
        z = self.bn_layers[0](z, training)
        for i in range(len(self.dense_layers)-1):
            z = self.dense_layers[i](z)
            z = self.bn_layers[i+1](z, training)                        # test after relu
            z = tf.nn.relu(z)
        z = self.dense_layers[-1](z)
        z = self.bn_layers[-1](z, training)

        return z
