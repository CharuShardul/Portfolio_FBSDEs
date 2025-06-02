
import numpy as np
import tensorflow as tf
import logging
import time
import munch
import json
import FBSDE_Parameters as FP

DELTA_CLIP = 50.0

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
        T_grid = dt*np.array([[t for t in range(self.eqn_config.t_grid_size)]
                             for _ in range(num_sample)])
        S = [self.S_init[i]*np.exp((self.b[i] - tf.reduce_sum(self.sig[i]**2)/2)*T_grid +
            tf.reduce_sum(self.sig[i][None, None, :]*W, axis=2, keepdims=False))
            for i in range(self.eqn_config.W_dim)]
        S = np.array([S[i]/(self.equation.gamma[0]*np.exp(self.eqn_config.r*T_grid) +
                   tf.reduce_sum(self.equation.gamma[1:, None, None]*S, axis=0)) for i in range(self.eqn_config.W_dim)])

        return W, S

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.Sample(self.net_config.valid_size)

        for p in range(self.fic_num):
            # begin sgd iteration
            for step in range(self.net_config.num_iterations + 1):
                #print("Step = ", step)
                if step % self.net_config.logging_frequency == 0:
                    #print((self.model.trainable_variables))
                    #print(self.model.summary)
                    loss = self.loss_fn(valid_data, training=False).numpy()
                    y_init = [self.y_init[0].numpy()[0], self.y_init[1].numpy()]
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

                    logging.info("fict_num: %3u,   step: %5u,   loss: %.4e,  Y0[0]: %.4e, Y0[1][0]: %.4e,  Y0[1][1]: %.4e,   elapsed time: %3u"
                                 % (p, step, loss, y_init[0], y_init[1][0], y_init[1][1], elapsed_time))

                    '''if step != 0:
                        # Save model checkpoint
                        saved_model = r'model_checkpoints\model_fict{}_iter{}'.format(p, step)
                        tf.keras.models.save_model(self.model, saved_model)
                        # Check saved model parameters
                        load_y0, load_y1 = self.model_check(saved_model)
                        logging.info("saved_model: fict_num: %3u,   step: %5u,   Y0[0]: %.4e,   Y0[1]: %.4e"
                                     % (p, step, load_y0, load_y1))'''
                self.train_step(self.Sample(self.net_config.batch_size))
                #print(p, '\t', step)

        return np.array(training_history)

    def loss_fn(self, input, training):
        #x_terminal, y_terminal = self.model(input, training)
        x0, x1, y0, y1 = self.model(input, training)
        #print("loss: y1_shape", tf.shape(y1))
        delta = tf.square(tf.abs(y0)) + tf.reduce_sum(tf.stack([tf.square(tf.abs(y1[i]))
                                                                for i in range(self.eqn_config.W_dim)]), axis=0)
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

        self.Exp_X = np.array([0.0 for _ in range(self.eqn_config.t_grid_size)])

        self.y_init_0 = tf.Variable(np.random.uniform(low=self.net_config.y_init_range[0],
                                                      high=self.net_config.y_init_range[1],
                                                      size=[1]))
        self.y_init_1 = tf.stack([tf.Variable(np.random.uniform(low=0.5,
                                                      high=0.8,
                                                      size=[1])) for _ in range(self.eqn_config.W_dim)])
        self.z_init_0 = tf.stack([tf.Variable(np.random.uniform(low=-0.1,
                                                      high=0.1,
                                                      size=[1])) for _ in range(self.eqn_config.W_dim)])
        self.z_init_1 = tf.stack([[tf.Variable(np.random.uniform(low=-0.1,
                                                      high=0.1,
                                                      size=[1])) for _ in range(self.eqn_config.W_dim)]
                         for _ in range(self.eqn_config.W_dim)])

        self.subnet = [Subnet(config) for _ in range(self.t_grid_size - 1)]

    def call(self, input, training):
        ''' S.shape = (W_dim, sample, time)
         W.shape = (sample, time, W_dim'''
        W, S = input
        dt = self.eqn_config.total_time/self.eqn_config.t_grid_size

        all_ones = tf.ones(shape=(tf.shape(W)[0], 1), dtype="float64")
        all_ones_N = tf.ones(shape=(tf.shape(W)[0], self.eqn_config.W_dim), dtype="float64")
        all_ones_N_N = tf.ones(shape=(tf.shape(W)[0], self.eqn_config.W_dim, self.eqn_config.W_dim), dtype="float64")
        #all_one_mat_2 = tf.Variable(tf.ones(shape=([tf.shape(W)[0, self.equation.X_dim]]), dtype="float64"))
        #all_one_mat_3 = tf.Variable(tf.ones(shape=tf.stack([tf.shape(W)[0, self.equation.X_dim]]), dtype="float64"))
        '''x0 = all_ones * x[0]
        x1 = all_ones * x[1]
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
        '''
        x = self.equation.X_init
        x0 = all_ones * x[0]
        x1 = [all_ones * x[1] for _ in range(self.eqn_config.W_dim)]
        y0 = all_ones * self.y_init_0
        #y1 = [[[self.y_init_1[i]] for i in range(self.eqn_config.W_dim)] for _ in range(tf.shape(W)[0])]
        y1 = tf.stack([all_ones * self.y_init_1[i] for i in range(self.eqn_config.W_dim)])
        #print("type:", y1[0])
        #print('y1', y1[0].shape)
        #z0 = tf.stack([tf.stack([self.z_init_0[i][0] for i in range(self.eqn_config.W_dim)]) for _ in range(tf.shape(W)[0])])
        #print("type:", z0[0, :])
        z0 = tf.stack(tf.concat([all_ones * self.z_init_0[i] for i in range(self.eqn_config.W_dim)], axis=1))
        #print('z0_shape', tf.shape(z0))
        # z1 = [tf.stack([[self.z_init_1[i][j][0] for j in range(self.eqn_config.W_dim)]
        #      for _ in range(tf.shape(W)[0])]) for i in range(self.eqn_config.W_dim)]
        z1 = [tf.stack(tf.concat([all_ones * self.z_init_1[i][j] for j in range(self.eqn_config.W_dim)], axis=1))
              for i in range(self.eqn_config.W_dim)]
        #print('z1_shape', tf.shape(z1))
        #z1 = [all_ones * self.y_init_1[i] for i in range(self.eqn_config.W_dim)]

        x = (x0, x1)
        y = (y0, y1)
        z = (z0, z1)
        '''x = tf.concat((x0, x1), axis=-1)
        print("x", x.shape)
        y = tf.concat((y0, y1), axis=-1)
        print("y", y.shape)
        z = tf.concat((z0, z1), axis=-1)
        print("z", z.shape)'''

        for t in range(self.eqn_config.t_grid_size - 1):

            x0 = x0 + (self.equation.aV(S[:, :, t]) * x0 +
                 sum([self.equation.ad(S[:, :, t])[i] * x1[i] for i in range(self.eqn_config.W_dim)]))*dt + \
                 tf.reduce_sum((self.equation.bV(S[:, :, t]) * x0 +
                                (sum([self.equation.bd(S[:, :, t])[i] * x1[i] for i in range(self.eqn_config.W_dim)])))*
                                (W[:, t + 1, :] - W[:, t, :]), axis=-1, keepdims=True)
            # if t == 0:
            # print("a = ", tf.shape(self.equation.f(t, x, y, z, S[:, t])[0]))
            # print("b = ", tf.shape(z0))
            # x = tf.stack((x0, x1), axis=1)
            #print('x0', x0.shape)
            x = (x0, x1)        #tf.stack((x0, x1), axis=1)
            y0 = y0 - self.equation.f(t, x, y, z, S[:, :, t])[0] * dt + tf.reduce_sum(z0 * (W[:, t + 1, :] - W[:, t, :]),
                                                                                      axis=1, keepdims=True)
            #print('y0', np.shape(self.equation.f(t, x, y, z, S[:, :, t])[0]))
            #print((tf.reduce_sum(z1[1, :, :] * (W[:, t + 1, :] - W[:, t, :]), axis=-1, keepdims=True)))
            #print(np.shape(y1[0] - self.equation.f(t, x, y, z, S[:, :, t])[1][0] * dt))
            #print(np.shape(y1[1]))

            y1 = [y1[i] - self.equation.f(t, x, y, z, S[:, :, t])[1][i] * dt + \
                  tf.reduce_sum(z1[i]*(W[:, t + 1, :] - W[:, t, :]), axis=-1, keepdims=True) \
                  for i in range(self.eqn_config.W_dim)]
            #print('y1_shape', tf.shape(y1), tf.shape(y1[1]))
            x1 = [x1[i] - y1[i] * dt for i in range(self.eqn_config.W_dim)]
            #print('x1shape', np.shape(x1))
            y = (y0, y1)
            z = tf.concat([x0, x1[0], x1[1], y0, y1[0], y1[1]], axis=-1)
            #print("z shape", tf.shape(z))
            z = self.subnet[t](z, training)
            #print("z shape", tf.shape(z))
            z0 = z[:, :2]
            #print('z0', tf.shape(z0))
            #z0[:, 1] = z[:, 1]
            z1[0] = z[:, 2:4]
            z1[1] = z[:, 4:]
            #z1[0, :] = z[:, 4]
            #z1[0, :] = z[:, 5]
            #z1 = tf.stack([z[:, 3], z[:, 4]])

            z = (z0, z1)
            #print("t", t)
            #print("z shape", type(z[1][:, 0, 0]), np.shape(z[1]))
            #print("t", t)
            # y = (y0, y1)
            # z = (z0, z1)
            # z0, z1 = z[:, 0, np.newaxis], z[:, 1, np.newaxis]
            #z0, z1 = tf.keras.layers.Lambda(lambda z: z[0, np.newaxis])(z), \
            #         tf.keras.layers.Lambda(lambda z: z[1, np.newaxis])(z)
        x0 = x0 + (self.equation.aV(S[:, :, t]) * x0 +
                 sum([self.equation.ad(S[:, :, t])[i] * x1[i] for i in range(self.eqn_config.W_dim)]))*dt + \
                 tf.reduce_sum((self.equation.bV(S[:, :, t]) * x0 +
                                (sum([self.equation.bd(S[:, :, t])[i] * x1[i] for i in range(self.eqn_config.W_dim)])))*
                                (W[:, -1, :] - W[:, -2, :]), axis=-1, keepdims=True)
        y0 = y0 - self.equation.f(t, x, y, z, S[:, :, t])[0] * dt + tf.reduce_sum(z0 * (W[:, -1, :] - W[:, -2, :]),
                                                                                      axis=1, keepdims=True)
        y1 = tf.stack([y1[i] - self.equation.f(t, x, y, z, S[:, :, t])[1][i] * dt +
                  tf.reduce_sum(z1[i]*(W[:, -1, :] - W[:, -2, :]), axis=-1, keepdims=True)
                  for i in range(self.eqn_config.W_dim)])
        x1 = [x1[i] - y1[i] * dt for i in range(self.eqn_config.W_dim)]

        #print("y:", tf.shape(y), "z:", tf.shape(z))
        #print("S shape =", tf.shape(tf.concat([S[:, 0], S[:, 0]], axis=1)))
        '''for t in range(self.eqn_config.t_grid_size -1):                         # optimize aV, bV, etc. calls.
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
        x1 = x1 - y1*dt'''

        return x0, x1, y0, y1


class Subnet(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        dim_0 = config.eqn_config.W_dim
        dim_1 = (config.eqn_config.W_dim)**2
        Z_dim = dim_0 + dim_1
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
