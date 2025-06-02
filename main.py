'''This code is part of the project "Long term dynamic portfolio optimization using infinite horizon McKean-Vlasov FBSDEs"
, which implements a global direct solver for forward-backward stochastic differential equations (FBSDEs). It uses 
feed-forward neural networks implemented in TensorFlow for numerical computations and is designed to solve FBSDEs 
with a specific configuration defined in the `FBSDE_Parameters_new` module. The code also includes functionality for
 logging, plotting results, and saving configurations.
 
 This code was developed as part of my PhD at University of Bordeaux under the supervision of Prof. Adrien Richou and 
 Prof. Emmanuel Gobet. Please refer to the PhD thesis of Charu Shardul (HAL link: https://theses.hal.science/tel-04627360v1) 
 for details.
 '''

import json
import munch
import os
import logging
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time
from targ_dist import TargDist

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np

import FBSDE_Parameters_new as equation
#from Global_direct_solver import FBSDEsolver
from Global_direct_solver import FBSDEsolver

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

    #true_y_init = config.eqn_config.explicit_solution
    
    # higher alpha values penalize the deviation from the target distribution more.
    alpha = [2.0]
    #alpha = [0.0, 1.0, 5.0, 10.0, 15.0]

    # Calling the __init__ function for FBSDE class from FBSDE_Parameters_new.
    FBSDE = [getattr(equation, config.eqn_config.eqn_name)(config.eqn_config, b_sig_dat, alpha[i])
             for i in range(len(alpha))]    
    #tf.keras.backend.set_floatx(config.net_config.dtype)

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
    #for i in range(len(alpha)):
    #   trained_models[i].save_weights(r"Numerical_experiments\Checkpoints\alpha_{}".format(alpha[i]))
    #test_time = time.time()
    #training_history = train[0]
    #trajec = train[1]
    trajec = [train[i][1] for i in range(len(alpha))]


    # Distance b/w measures

    #k_arr = config.eqn_config.nu_support
    X = [trajec[i][0] for i in range(len(alpha))]
    #print(X[0].shape)
    #pi = TargDist()
    #k = np.linspace(0.92, 1.08, 11)
    #p = [[pi.p(k=k[i], T=(t / config.eqn_config.t_grid_size)) for i in range(len(k))] for t in range(config.eqn_config.t_grid_size)]
    #distance = np.mean([np.maximum(np.zeros(len(X[0][0])), np.mean(np.maximum(np.zeros_like(X[0]), k - X[0]), axis=0) - p[:, k])
    #                    for k in k_arr], axis=0)
    distance = np.array([FBSDE[0].dist(t, X[0][:, t]) for t in range(int(config.eqn_config.total_time*config.eqn_config.t_grid_size))])
    #print("distance shape ", distance.shape)

    '''
    # CDF of terminal wealth

    X_end = [X[i][:, -1] for i in range(len(alpha))]
    sort = [np.sort(X_end[i]) for i in range(len(alpha))]
    length = config.net_config.valid_size
    wealth_axis = np.linspace(0.85, 1.15, 61)
    cdf = [np.array([np.count_nonzero(sort[j] <= wealth_axis[i])/length for i in range(len(wealth_axis))])
           for j in range(len(alpha))]
    fig, axs = plt.subplots(figsize=(18, 21), dpi=75)
    for i in range(len(alpha)):
        axs.plot(wealth_axis, cdf[i], label=alpha[i])
    plt.legend()
    axs.set_title("CDF for different alpha")
    fig.savefig(r"Numerical_experiments\CDF\alpha_0_test")
    plt.show()
    '''
    test_time = time.time()
    fig, axs = plt.subplots(2, 2, figsize=(18, 21), dpi=75)
    # Plotting averages
    x_axis = np.arange(0, int(config.eqn_config.total_time*config.eqn_config.t_grid_size), 1)

    axs[0, 0].plot(x_axis, np.mean(trajec[-1][0], axis=0))
    axs[0, 0].set_title("Wealth mean")
    axs[0, 1].plot(x_axis, -np.mean(trajec[-1][3], axis=0))
    axs[0, 1].set_title("Trading speed mean")
    axs[1, 0].plot(x_axis, np.mean(trajec[-1][1], axis=0))
    axs[1, 0].set_title("Mean portfolio process")
    axs[1, 1].plot(x_axis, np.mean(trajec[-1][2], axis=0))
    axs[1, 1].set_title("E[Y^0_t]")

    plt.show()
    
    for a in alpha:
        fig.savefig(r"Numerical_experiments\Global_direct_solver\Means\alpha_{}.png".format(a))

    fig, axs = plt.subplots(4, 2, figsize=(24, 21), dpi=75)
    
    # Plotting some sample trajectories
    for s in range(6):
        axs[0, 0].plot(x_axis, trajec[0][0][s])
        axs[0, 0].set_title("Wealth trajectories")
        axs[0, 1].plot(x_axis, trajec[0][2][s])
        axs[0, 1].set_title("Y^0_t")
        axs[1, 0].plot(x_axis, -trajec[0][3][s])
        axs[1, 0].set_title("Trading speed trajectories")
        axs[1, 1].plot(x_axis, trajec[0][1][s])
        axs[1, 1].set_title("Portfolio process trajectories")
        axs[2, 0].plot(x_axis, trajec[0][4][s])
        axs[2, 0].set_title("Z^0_t")
        axs[2, 1].plot(x_axis, trajec[0][5][s])
        axs[2, 1].set_title("Z^1_t")
        axs[3, 0].plot(x_axis, trajec[0][6][s])
        axs[3, 0].set_title("L-derivative trajectories")
    axs[3, 1].plot(x_axis, distance)
    axs[3, 1].set_title("Distance b/w measures")

    test_time = time.time() - test_time
    print("Plotting time:", test_time)

    plt.show()

    for a in alpha:
        fig.savefig(r"Numerical_experiments\Global_direct_solver\Trajecs\alpha_{}.png".format(a))
    #plt.legend()


if __name__ == '__main__':
    app.run(main)
