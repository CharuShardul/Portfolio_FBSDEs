
import numpy as np
import tensorflow as tf
import logging
import time
import munch
import json
import FBSDE_Parameters as FP

#DELTA_CLIP = 50.0

'''conf_path = "configs\FBSDE_config.json"
stock_data = r'Data_files\b_and_sig.json'
with open(conf_path) as json_config, open(stock_data) as json_stock_dat:
    config = json.load(json_config)
    b_sig_dat = json.load(json_stock_dat)

config = munch.munchify(config)
b_sig_dat = munch.munchify(b_sig_dat)
equation = FP.FBSDE'''

class FBSDEsolver(object):
    def __init__(self, config, equation):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.equation = equation
        self.model = fullNN(equation, config)
        # Drift and volatility are yearly in "equation", r is already yearly.
        self.b = equation.b
        self.sig = equation.sig

        self.S_init = equation.S_init
        self.y_init = [self.model.y_init_0, self.model.y_init_1]
        self.fic_num = self.eqn_config.fict_play_num
        self.lamb = self.eqn_config.lamb

        #lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #    self.net_config.lr_boundaries, self.net_config.lr_values)
        #lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #        self.net_config.lr_boundaries, self.net_config.lr_values)
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    self.lr_schedule()[0], self.lr_schedule()[1])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)

    def lr_schedule(self):
        lr_boundaries = np.linspace(0, self.fic_num * self.net_config.num_iterations, self.fic_num*10 + 1,
                                    dtype=int)[1:].tolist()
        lr_values = np.array([])
        for p in range(self.fic_num):
            temp = np.array([5*1e-2*(0.5**i) for i in range(10)])
            lr_values = np.append(lr_values, temp)
        lr_values = np.append(lr_values, 0.4*lr_values[-1]).tolist()
        print("bound:", lr_boundaries)
        print("values:", lr_values)
        return lr_boundaries, lr_values

    def Sample(self, num_sample):
        # Generating mini-batch samples for Brownian motion and relative stock price process
        #self.b = 0
        dt = self.equation.delta_t
        W = np.random.normal(0, np.sqrt(dt), size=(num_sample, self.eqn_config.t_grid_size - 1, self.eqn_config.W_dim))
        zero_matrix = np.zeros(shape=[num_sample, 1, self.eqn_config.W_dim], dtype='float64')
        W = np.concatenate([zero_matrix, W], axis=1)
        W = np.cumsum(W, axis=1)                        # Full brownian motion as a tensor
        T_grid = dt*np.array([[t for t in range(self.eqn_config.t_grid_size)]
                             for _ in range(num_sample)])
        #S = [self.S_init[i]*np.exp((self.b[i] - tf.reduce_sum(self.sig[i]**2)/2)*T_grid +
        #    tf.reduce_sum(self.sig[i][None, None, :]*W, axis=2, keepdims=False))
        #    for i in range(self.eqn_config.W_dim)]
        S = self.S_init[0]*np.exp((self.b[0] - (self.sig[0, 0]**2)/2)*T_grid + self.sig[0, 0]*W[:, :, 0])
        #S = np.array([S[i]/(self.equation.gamma[0]*np.exp(self.eqn_config.r*T_grid) +
        #           tf.reduce_sum(self.equation.gamma[1:, None, None]*S, axis=0)) for i in range(self.eqn_config.W_dim)])
        #print("S:1", S.shape)
        S = S / (self.equation.gamma[0] * np.exp(self.eqn_config.r * T_grid) + self.equation.gamma[1]*S)
        #print("S:2", S.shape)

        return W, S

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.Sample(self.net_config.valid_size)

        for step in range(self.net_config.num_iterations + 1):
            if step % self.net_config.logging_frequency == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                y_init = [self.y_init[0].numpy(), self.y_init[1].numpy()]
                #theta = self.model.theta.numpy()
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init[0], y_init[1], elapsed_time])
                logging.info("step: %5u,   loss: %.4e,  Y0[0]: %.4e, Y0[1]: %.4e, elapsed time: %3u"
                             % (step, loss, y_init[0], y_init[1], elapsed_time))
            self.train_step(self.Sample(self.net_config.batch_size))

        X0, X1, Y0, Y1, Z0, Z1, ldx_arr = self.model(valid_data, training=False, train_end=True)
        print("shapes=", X0.shape, X1.shape, Y0.shape, Y1.shape, Z0.shape, Z1.shape, ldx_arr.shape)
        trajec = (X0, X1, Y0, Y1, Z0, Z1, ldx_arr)
        return training_history, trajec

    def loss_fn(self, input, training):
        #x_terminal, y_terminal = self.model(input, training)
        x0, x1, y0, y1 = self.model(input, training)
        beta = 1.0
        #print("loss: y_shape", tf.shape(y0), tf.shape(y1))
        delta = tf.square(tf.abs(y0)) + beta*tf.square(tf.abs(y1))#beta*tf.reduce_sum(tf.stack([tf.square(tf.abs(y1[i]))
                                                                #for i in range(self.eqn_config.W_dim)]), axis=0)
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

    def model_check(self, model_dir):
        model = tf.keras.models.load_model(model_dir)
        #model.compile()
        y_0 = model.y_init_0.numpy()[0]
        y_1 = model.y_init_1.numpy()[0]
        valid_data = self.Sample(self.net_config.valid_size)
        #loss = self.loss_fn(model, valid_data, training=False).numpy()
        return y_0, y_1




class fullNN(tf.keras.Model):
    def __init__(self, equation, config):
        super(fullNN, self).__init__()
        self.equation = equation
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.t_grid_size = config.eqn_config.t_grid_size
        self.lamb = self.eqn_config.lamb
        #self.zeta = tf.TensorArray(dtype=tf.float64, size=len(self.eqn_config.nu_support), clear_after_read=False)
        self.k = self.eqn_config.nu_support
        #for i in range(len(self.k)):
        #    self.zeta.write(i, tf.cast(tf.math.log(1.0 + tf.exp(self.lamb * (self.k[i] - 1.0))), tf.float64))

        self.y_init_0 = tf.Variable(np.random.uniform(low=self.net_config.y_init_range[0],
                                                      high=self.net_config.y_init_range[1],
                                                      size=[1]))
        self.y_init_1 = tf.Variable(np.random.uniform(low=-0.5,
                                                      high=0.5,
                                                      size=[1]))


        #self.z_init_0 = tf.Variable(np.random.uniform(low=-0.5,
        #                                              high=0.5,
        #                                              size=[1]))
        #self.z_init_1 = tf.Variable(np.random.uniform(low=-0.5,
        #                                              high=0.5,
        #                                              size=[1]))


        self.subnet = Subnet(config)  # for _ in range(self.t_grid_size - 2)]

    def call(self, input, training, train_end=False):
        ''' S.shape = (W_dim, sample, time)
         W.shape = (sample, time, W_dim)'''
        W, S = input
        dt = self.eqn_config.total_time/self.eqn_config.t_grid_size
        epsilon = 0.1

        all_ones = tf.ones(shape=(tf.shape(W)[0], 1), dtype="float64")

        self.zeta = np.array([np.log(1 + np.e ** (self.lamb * (self.k[i] - np.ones(W.shape[0])))) for i in range(len(self.k))])

        x_init = self.equation.X_init
        x0 = all_ones * x_init[0]
        x1 = all_ones * x_init[1]
        y0 = all_ones * self.y_init_0
        y1 = all_ones * self.y_init_1
        z = self.subnet(all_ones * 0.0, tf.concat([x0, x1, y0, y1], axis=1))
        z0 = z[:, 0, None]
        z1 = z[:, 0, None]


        y = (y0, y1)
        z = (z0, z1)
        '''x = tf.concat((x0, x1), axis=-1)
        print("x", x.shape)
        y = tf.concat((y0, y1), axis=-1)
        print("y", y.shape)
        z = tf.concat((z0, z1), axis=-1)
        print("z", z.shape)'''

        if train_end==True:
            x_star_0 = x0
            x_star_1 = x1
            y_star_0 = y0
            y_star_1 = y1
            z_star_0 = z0
            z_star_1 = z1
            ldx_arr = self.equation.ldx(0, x0, self.zeta)
            print("shape 0:", x_star_0.shape, x_star_1.shape, y_star_0.shape, y_star_1.shape, z_star_0.shape,
                  z_star_1.shape, ldx_arr.shape)

        for t in range(self.eqn_config.t_grid_size - 2):

            x0 = x0 + (self.equation.aV(S[:, t, None]) * x0 +
                       self.equation.ad(S[:, t, None]) * x1) * dt + \
                 (self.equation.bV(S[:, t, None]) * x0 + \
                  self.equation.bd(S[:, t, None]) * x1) * (W[:, t + 1, :] - W[:, t, :])
            x1 = x1 - 50.0 * y1 * dt
            #print("X0", x0.shape)
            x = (x0, x1)  # tf.stack((x0, x1), axis=1)
            #print("Y1", y0.shape)
            y0 = y0 - self.equation.f(t, x, y, z, S[:, t, None], self.zeta)[0] * dt + z0 * (
                        W[:, t + 1, :] - W[:, t, :])
            #print("Y2", y0.shape)
            y1 = y1 - self.equation.f(t, x, y, z, S[:, t, None], self.zeta)[1] * dt + \
                 z1 * (W[:, t + 1, :] - W[:, t, :])
            # print('y1_shape', tf.shape(y1), tf.shape(y1[1]))
            # print('x1shape', np.shape(x1))
            y = (y0, y1)

            #print("z shape 0: ", tf.shape(z))
            z = self.subnet(all_ones*t*dt, tf.concat([x0, x1, y0, y1], axis=-1), training)
            z0, z1 = (z[:, 0, None], z[:, 1, None])
            z = (z0, z1)
            # print("z shape 2: ", z0.shape, z1.shape)

            self.zeta = tf.stack([tf.math.log(1.0 + tf.exp(self.lamb * (k - x0)))
                                     for k in self.eqn_config.nu_support], axis=0)
            #print("zeta shape: ", self.zeta.shape)
            #self.zeta = np.array([tf.reduce_mean(tf.math.log(1.0 + tf.exp(self.lamb * (k - x0))))
            #                         for k in self.eqn_config.nu_support], axis=0)
            #for i in range(len(self.k)):
            #    self.zeta.write(i, tf.reduce_mean(tf.math.log(1.0 + tf.exp(self.lamb * (self.k[i] - x0)))))


            if train_end == True:
                x_star_0 = tf.concat([x_star_0, x0], axis=-1)
                x_star_1 = tf.concat([x_star_1, x1], axis=-1)
                y_star_0 = tf.concat([y_star_0, y0], axis=-1)
                y_star_1 = tf.concat([y_star_1, y1], axis=-1)
                z_star_0 = tf.concat([z_star_0, z0], axis=-1)
                z_star_1 = tf.concat([z_star_1, z1], axis=-1)
                ldx_arr = tf.concat([ldx_arr, self.equation.ldx(t, x0, self.zeta)], axis=-1)


        x0 = x0 + (self.equation.aV(S[:, t+1, None]) * x0 +
                   self.equation.ad(S[:, t+1, None]) * x1) * dt + \
             (self.equation.bV(S[:, t+1, None]) * x0 + \
              self.equation.bd(S[:, t+1, None]) * x1) * (W[:, t+2, :] - W[:, t+1, :])
        # print("X0", x0.shape)
        x1 = x1 - 50.0 *y1 * dt  # + epsilon * (W[:, t + 1, :] - W[:, t, :])
        x = (x0, x1)  # tf.stack((x0, x1), axis=1)
        # print("Y1", y0.shape)
        y0 = y0 - self.equation.f(t+1, x, y, z, S[:, t+1, None], self.zeta)[0] * dt + z0 * (
                W[:, t+2, :] - W[:, t+1, :])
        # print("Y2", y0.shape)
        y1 = y1 - self.equation.f(t+1, x, y, z, S[:, t+1, None], self.zeta)[1] * dt + \
             z1 * (W[:, t+2, :] - W[:, t+1, :])
        # print('y1_shape', tf.shape(y1), tf.shape(y1[1]))
        self.zeta = tf.stack([tf.math.log(1.0 + tf.exp(self.lamb * (k - x0)))
                                 for k in self.eqn_config.nu_support], axis=0)

        if train_end == True:
            x_star_0 = tf.concat([x_star_0, x0], axis=-1)
            x_star_1 = tf.concat([x_star_1, x1], axis=-1)
            y_star_0 = tf.concat([y_star_0, y0], axis=-1)
            y_star_1 = tf.concat([y_star_1, y1], axis=-1)
            z_star_0 = tf.concat([z_star_0, z0], axis=-1)
            z_star_1 = tf.concat([z_star_1, z1], axis=-1)
            ldx_arr = tf.concat([ldx_arr, self.equation.ldx(t+1, x0, self.zeta)], axis=-1)
            #print("test 2")
            #for i in range(len(self.k)):
            #    self.zeta.write(i, tf.reduce_mean(tf.math.log(1.0 + tf.exp(self.lamb * (self.k[i] - x0)))))


        if train_end == False:
            return x0, x1, y0, y1
        elif train_end == True:
            return x_star_0.numpy()[:, :], x_star_1.numpy()[:, :], y_star_0.numpy()[:, :], y_star_1.numpy()[:, :],\
                   z_star_0.numpy()[:, :], z_star_1.numpy()[:, :], ldx_arr


class Subnet(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        dim_0 = config.eqn_config.W_dim
        dim_1 = (config.eqn_config.W_dim)**2
        Z_dim = dim_0 + dim_1
        num_hiddens = [50 + config.eqn_config.X_dim, 50 + config.eqn_config.X_dim]
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

    def call(self, t, z, training):
        #z = tf.concat([x, y], axis=1)
        #z = tf.concat([x0, y0], axis=1)
        #z = self.bn_layers[0](z, training)
        z = tf.concat([t, z], axis=1)
        for i in range(len(self.dense_layers)-1):
            z = self.dense_layers[i](z)
            #z = self.bn_layers[i+1](z, training)                        # test after relu
            z = tf.nn.relu(z)
        z = self.dense_layers[-1](z)
        #z = self.bn_layers[-1](z, training)

        return z
