import json
import munch
import os
import logging
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np

import FBSDE_Parameters as equation
from FBSDE_NN_1d import FBSDEsolver

flags.DEFINE_string('config_path', r'configs\FBSDE_config_1d.json',
                    "The path to load config file.")

flags.DEFINE_string('stock_data', r'Data_files\b_and_sig.json',
                    "The path to stock(s) parameters.")

flags.DEFINE_string('exp_name', 'test',
                    "The name of numerical experiments, prefix for logging")

FLAGS = flags.FLAGS
FLAGS.log_dir = './logs'  # directory where to write event logs and output array


def main(argv):
    del argv

    with open(FLAGS.config_path) as json_config, open(FLAGS.stock_data) as json_stock_dat:
        config = json.load(json_config)
        b_sig_dat = json.load(json_stock_dat)
    config = munch.munchify(config)
    b_sig_dat = munch.munchify(b_sig_dat)

    true_y_init = config.eqn_config.explicit_solution
    #alpha = [0.0, 1.0, 5.0, 10.0, 15.0]
    alpha = [1.0]
    FBSDE = [getattr(equation, config.eqn_config.eqn_name)(config.eqn_config, b_sig_dat, alpha[i])
             for i in range(len(alpha))]    # Calling the __init__ function.
    tf.keras.backend.set_floatx(config.net_config.dtype)

    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                       for name in dir(config) if not name.startswith('__')),
                  outfile, indent=2)

    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)


    fbsde_solver = [FBSDEsolver(config, FBSDE[i]) for i in range(len(alpha))]
    train = [fbsde_solver[i].train() for i in range(len(alpha))]
    trained_models = [fbsde_solver[i].model for i in range(len(alpha))]
    for i in range(len(alpha)):
        trained_models[i].save_weights(r"Numerical_experiments\Checkpoints\alpha_{}".format(alpha[i]))
    #test_time = time.time()
    #training_history = train[0]
    #trajec = train[1]
    trajec = [train[i][1] for i in range(len(alpha))]


    '''
    # L-derivative
    X = trajec[-1][0]
    zeta = trajec[-1][6]
    print("X and zeta shape:", X.shape, zeta.shape)
    Ldx = np.array([[FBSDE.ldx(t=j, x0=X[i, j], zeta=zeta[j]) for j in range(config.eqn_config.t_grid_size)]
            for i in range(config.net_config.valid_size)])
    print("Ldx shape:", Ldx.shape)'''

    # Distance b/w measures

    X = [trajec[i][-1][0] for i in range(len(alpha))]
    #distance = [FBSDE.dist(t=j, x0=X[:, j]) for j in range(len(X[0]))]

    # CDF of terminal wealth

    X_term = [X[i][:, -1] for i in range(len(alpha))]
    sort = [np.sort(X_term[i]) for i in range(len(alpha))]
    length = config.net_config.valid_size
    wealth_axis = np.linspace(0.85, 1.15, 121)
    cdf = [np.array([np.count_nonzero(sort[j] <= wealth_axis[i])/length for i in range(len(wealth_axis))])
           for j in range(len(alpha))]
    fig, axs = plt.subplots(figsize=(18, 21), dpi=75)
    for i in range(len(alpha)):
        axs.plot(wealth_axis, cdf[i], label=alpha[i])
    plt.legend()
    axs.set_title("CDF for different alpha")
    fig.savefig(r"Numerical_experiments\CDF\alpha_0_test")
    plt.show()


    '''if true_y_init == True:
        np.savetxt('{}_training_history.csv'.format(path_prefix),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%.4e', '%d'],
               delimiter=",",
               header='step, loss_function, target_value, relative_error, elapsed_time',
               comments='')
    else:
        np.savetxt('{}_training_history.csv'.format(path_prefix),
                   training_history,
                   fmt=['%d', '%d', '%.5e', '%.5e', '%.5e', '%d'],
                   delimiter=",",
                   header='fict_num, step, loss_function, Y_0^1, Y_0^2, elapsed_time',
                   comments='')
    '''

    '''
    fig, axs = plt.subplots(2, 2, figsize=(18, 21), dpi=75)
    x_axis = np.arange(0, config.eqn_config.t_grid_size, 1)
    #rho_axis = np.linspace(0.1, 10.0, 10)
    # Plotting fictitious play iterations of means
    p = 0
    for p in range(config.eqn_config.fict_play_num-1):
        axs[0, 0].plot(x_axis, np.mean(trajec[p][0], axis=0), '--', label=p+1)
        axs[0, 1].plot(x_axis, -np.mean(trajec[p][3], axis=0), '--', label=p+1)
        axs[1, 0].plot(x_axis, np.mean(trajec[p][1], axis=0), '--', label=p+1)
        axs[1, 1].plot(x_axis, np.mean(trajec[p][2], axis=0), '--', label=p+1)
        p += 1
    axs[0, 0].plot(x_axis, np.mean(trajec[-1][0], axis=0), label=p+2)
    axs[0, 0].set_title("Wealth mean")
    axs[0, 1].plot(x_axis, -np.mean(trajec[-1][3], axis=0), label=p+2)
    axs[0, 1].set_title("Trading speed mean")
    axs[1, 0].plot(x_axis, np.mean(trajec[-1][1], axis=0), label=p+2)
    axs[1, 0].set_title("Mean portfolio process")
    axs[1, 1].plot(x_axis, np.mean(trajec[-1][2], axis=0), label=p+2)
    axs[1, 1].set_title("E[Y^0_t]")
    test_time = time.time() - test_time
    print("Plotting time:", test_time)
    plt.show()
    fig.savefig(r"Numerical_experiments\Means\alpha_16_lambtil_8_beta_5_noBN")


    fig, axs = plt.subplots(4, 2, figsize=(24, 21), dpi=75)
    # Plotting trajectories
    p = config.eqn_config.fict_play_num
    for s in range(int(len(trajec[-1][0]) / 32)):
        axs[0, 0].plot(x_axis, trajec[-1][0][s], label=p)
        axs[0, 0].set_title("Wealth trajectories")
        axs[0, 1].plot(x_axis, -trajec[-1][3][s], label=p)
        axs[0, 1].set_title("Trading speed trajectories")
        axs[1, 0].plot(x_axis, trajec[-1][1][s], label=p)
        axs[1, 0].set_title("Portfolio process trajectories")
        axs[1, 1].plot(x_axis, trajec[-1][2][s], label=p)
        axs[1, 1].set_title("Y^0_t")
        axs[2, 0].plot(x_axis, trajec[-1][4][s], label=p)
        axs[2, 0].set_title("Z^0_t")
        axs[2, 1].plot(x_axis, trajec[-1][5][s], label=p)
        axs[2, 1].set_title("Z^1_t")

        axs[3, 0].plot(x_axis, Ldx[s])
        axs[3, 0].set_title("L-derivative trajectories")

    axs[3, 1].plot(x_axis, distance)
    axs[3, 1].set_title("Distance b/w measures")

    plt.show()
    fig.savefig(r"Numerical_experiments\Trajecs\alpha_16_lambtil_8_beta_5_noBN")
    #plt.legend()
    '''


if __name__ == '__main__':
    app.run(main)
