import numpy as np
import tensorflow as tf
import logging
import time
import munch
import json
import FBSDE_Parameters_new as FP

#DELTA_CLIP = 50.0

'''
conf_path = "configs\FBSDE_config.json"
stock_data = r'Data_files\b_and_sig.json'
with open(conf_path) as json_config, open(stock_data) as json_stock_dat:
    config = json.load(json_config)
    b_sig_dat = json.load(json_stock_dat)

config = munch.munchify(config)
b_sig_dat = munch.munchify(b_sig_dat)
equation = FP.FBSDE
'''


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
        #self.y_init = [self.model.y_init_0, self.model.y_init_1]
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
        n_lr = 11
        lr_boundaries = np.linspace(0, self.net_config.num_iterations, n_lr,
                                    dtype=int)[1:].tolist()
        lr_values = np.array([5*1e-3*(0.6**i) for i in range(n_lr)]).tolist()
        
        print("lr_bounds:", np.round(lr_boundaries, 6))
        print("lr_values:", np.round(lr_values, 6))
        return lr_boundaries, lr_values

    def Sample(self, num_sample):
        # Generating mini-batch samples for Brownian motion and relative stock price process
        dt = self.equation.delta_t
        W = np.random.normal(0, np.sqrt(dt), size=(num_sample, int(self.eqn_config.total_time*self.eqn_config.t_grid_size) - 1, self.eqn_config.W_dim))
        zero_matrix = np.zeros(shape=[num_sample, 1, self.eqn_config.W_dim], dtype='float32')
        W = np.concatenate([zero_matrix, W], axis=1)
        W = np.cumsum(W, axis=1)                        # Full brownian motion as a tensor
        T_grid = dt*np.array([[t for t in range(int(self.eqn_config['total_time']*self.eqn_config['t_grid_size']))]
                             for _ in range(num_sample)])
        S = self.S_init[0]*np.exp((self.b[0] - (self.sig[0, 0]**2)/2)*T_grid + self.sig[0, 0]*W[:, :, 0])
        S = S / (self.equation.gamma[0] * np.exp(self.eqn_config.r * T_grid) + self.equation.gamma[1]*S)
        

        return W, S

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.Sample(self.net_config.valid_size)

        '''for step in range(5):  # just a few steps for debugging
            print("Before:", self.model.y_init_0.numpy())
            self.train_step(self.Sample(self.net_config.batch_size))
            print("After:", self.model.y_init_0.numpy())'''

        for step in range(self.net_config.num_iterations + 1):
            if step % self.net_config.logging_frequency == 0:
                #print("test-lambda", self.eqn_config.lamb)
                loss = self.loss_fn(valid_data, training=False).numpy()
                y_init = [self.model.y_init_0.numpy(), self.model.y_init_1.numpy()]
                #theta = self.model.theta.numpy()
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init[0], y_init[1], elapsed_time])
                logging.info("step: %5u,   loss: %.4e,  Y0[0]: %.4e, Y0[1]: %.4e, elapsed time: %3u"
                             % (step, loss, y_init[0], y_init[1], elapsed_time))
            self.train_step(self.Sample(self.net_config.batch_size))

        X0, X1, Y0, Y1, Z0, Z1, ldx_arr = self.model(valid_data, training=False, train_end=True)
        #print("shapes=", X0.shape, X1.shape, Y0.shape, Y1.shape, Z0.shape, Z1.shape, ldx_arr.shape)
        trajec = (X0, X1, Y0, Y1, Z0, Z1, ldx_arr)
        return training_history, trajec

    def loss_fn(self, input, training):
        x0, x1, y0, y1 = self.model(input, training=training)
        # parameter for adjusting the weight of trading speeds in the cost functional
        beta = 1.0
        delta = tf.square(tf.abs(y0)) + beta*tf.square(tf.abs(y1))
        loss = tf.reduce_mean(delta)
        #loss += tf.square(self.model.y_init_0)
        #loss = tf.reduce_mean(tf.square(delta))

        return loss

    def grad(self, input, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(input, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        clipped_gradients = [tf.clip_by_value(gr, -5.0, 5.0) for gr in grad]
        #print("Gradients for y_init_0:", grad[0].numpy())
        del tape
        return clipped_gradients

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        # Get current learning rate from the optimizer's schedule
        #if hasattr(self.optimizer.learning_rate, '__call__'):
            # If using a schedule, pass in the optimizer's iteration count
        #    lr = self.optimizer.learning_rate(self.optimizer.iterations).numpy()
        #else:
        #    lr = self.optimizer.learning_rate.numpy()
        #print("Current learning rate:", lr)
        #print("Grad for y_init_0:", grad[0].numpy())
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        #print("y_init_0 after apply_gradients:", self.model.y_init_0.numpy())
        #print("Trainable variables:")
        #for v in self.model.trainable_variables:
        #    print(v.name, v.shape)



class fullNN(tf.keras.Model):
    def __init__(self, equation, config):
        super(fullNN, self).__init__()
        self.equation = equation
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.t_grid_size = config.eqn_config.t_grid_size
        self.lamb = config.eqn_config.lamb
        self.k = equation.k

        # Use add_weight so Keras tracks these as trainable variables!
        self.y_init_0 = self.add_weight(
            name="y_init_0",
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(
                minval=config.net_config.y_init_range[0],
                maxval=config.net_config.y_init_range[1]
            ),
            trainable=True,
            dtype=tf.float32
        )
        self.y_init_1 = self.add_weight(
            name="y_init_1",
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-0.5,
                maxval=0.5
            ),
            trainable=True,
            dtype=tf.float32
        )

        self.subnet = Subnet(config)

    def call(self, input, training, train_end=False):
        ''' S.shape = (W_dim, sample, time)
         W.shape = (sample, time, W_dim)'''
        W, S = input
        dt = 1/self.eqn_config['t_grid_size']
        epsilon = 0.1

        all_ones = tf.ones(shape=(tf.shape(W)[0], 1), dtype="float32")

        self.zeta = np.array([np.log(1 + np.e ** (self.lamb * (self.k[i] - np.ones(W.shape[0])))) for i in range(len(self.k))])

        x_init = self.equation.X_init
        x0 = all_ones * x_init[0]
        x1 = all_ones * x_init[1]
        y0 = all_ones * self.y_init_0
        y1 = all_ones * self.y_init_1
        #z = self.subnet(all_ones * 0.0, S[:, 0, None], tf.concat([x0, x1, y0, y1], axis=1))
        z = self.subnet(all_ones * 0.0, S[:, 0, None], tf.concat([x0, x1], axis=1), training=training)
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
            #print("shape 0:", x_star_0.shape, x_star_1.shape, y_star_0.shape, y_star_1.shape, z_star_0.shape,
            #      z_star_1.shape, ldx_arr.shape)

        for t in range(int(self.eqn_config['total_time']*self.eqn_config['t_grid_size']) - 2):

            x0 = x0 + (self.equation.aV(S[:, t, None]) * x0 +
                       self.equation.ad(S[:, t, None]) * x1) * dt + \
                 (self.equation.bV(S[:, t, None]) * x0 + \
                  self.equation.bd(S[:, t, None]) * x1) * (W[:, t + 1, :] - W[:, t, :])
            x1 = x1 - y1 * dt
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
            #z = self.subnet(all_ones*t*dt, tf.concat([x0, x1, y0, y1], axis=-1), training)
            #z = self.subnet(all_ones * t * dt, S[:, t, None], tf.concat([x0, x1, y0, y1], axis=-1), training)
            z = self.subnet(all_ones * t * dt, S[:, t, None], tf.concat([x0, x1], axis=-1), training=training)
            z0, z1 = (z[:, 0, None], z[:, 1, None])
            z = (z0, z1)
            # print("z shape 2: ", z0.shape, z1.shape)

            #self.zeta = tf.stack([tf.math.log(1.0 + tf.exp(self.lamb * (k - x0)))
            #                         for k in self.eqn_config.nu_support], axis=0)
            self.zeta = tf.stack([tf.math.log(1.0 + tf.exp(self.lamb * (k - x0)))
                                  for k in self.equation.k], axis=0)
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
        x1 = x1 - y1 * dt  # + epsilon * (W[:, t + 1, :] - W[:, t, :])
        x = (x0, x1)  # tf.stack((x0, x1), axis=1)
        # print("Y1", y0.shape)
        y0 = y0 - self.equation.f(t+1, x, y, z, S[:, t+1, None], self.zeta)[0] * dt + z0 * (
                W[:, t+2, :] - W[:, t+1, :])
        # print("Y2", y0.shape)
        y1 = y1 - self.equation.f(t+1, x, y, z, S[:, t+1, None], self.zeta)[1] * dt + \
             z1 * (W[:, t+2, :] - W[:, t+1, :])
        # print('y1_shape', tf.shape(y1), tf.shape(y1[1]))
        #self.zeta = tf.stack([tf.math.log(1.0 + tf.exp(self.lamb * (k - x0)))
        #                         for k in self.eqn_config.nu_support], axis=0)
        self.zeta = tf.stack([tf.math.log(1.0 + tf.exp(self.lamb * (k - x0)))
                                 for k in self.equation.k], axis=0)

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
        super(Subnet, self).__init__()
        dim_0 = config.eqn_config.W_dim
        dim_1 = (config.eqn_config.W_dim)**2
        Z_dim = dim_0 + dim_1
        num_hiddens = [10 + config.eqn_config.X_dim, 10 + config.eqn_config.X_dim]
        #self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(num_hiddens)+1)]
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
                                                   activation=None, #'relu',
                                                   kernel_initializer=None)#'he_normal')
                             for i in range(len(num_hiddens))
                             ]
        self.dense_layers.append(tf.keras.layers.Dense(Z_dim, activation=None))

    def build(self, input_shape):
        pass

    def call(self, t, s, z, training):
        #z = tf.concat([x, y], axis=1)
        #z = tf.concat([x0, y0], axis=1)
        #z = self.bn_layers[0](z, training=training)
        z = tf.concat([t, s, z], axis=1)
        for i in range(len(self.dense_layers)-1):
            z = self.dense_layers[i](z)
            #z = self.bn_layers[i+1](z, training=training)                        # test after relu
            z = tf.nn.relu(z)
        z = self.dense_layers[-1](z)
        #z = self.bn_layers[-1](z, training)

        return z


