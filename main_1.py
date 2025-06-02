import json
import munch
import os
import logging
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np

import FBSDE_Parameters as equation
from FBSDE_NN_1 import FBSDEsolver

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
    FBSDE = getattr(equation, config.eqn_config.eqn_name)(config.eqn_config, b_sig_dat)        # Calling the __init__ function.
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


    fbsde_solver = FBSDEsolver(config, FBSDE)
    train = fbsde_solver.train()
    training_history = train[0]
    trajec = train[1]
    if true_y_init == True:
        np.savetxt('{}_training_history.csv'.format(path_prefix),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%.4e', '%d'],
               delimiter=",",
               header='step, loss_function, target_value, relative_error, elapsed_time',
               comments='')
    else:
        np.savetxt('{}_training_history.csv'.format(path_prefix),
                   training_history,
                   fmt=['%d', '%d', '%.5e', '%.5e', '%d'],
                   delimiter=",",
                   header='fict_num, step, loss_function, Y_0, elapsed_time',
                   comments='')


    fig, axs = plt.subplots(1, 3, figsize=(15, 21), dpi=75)
    x_axis = np.arange(0, config.eqn_config.t_grid_size, 1)
    #rho_axis = np.linspace(0.1, 10.0, 10)
    # Plotting fictitious play iterations of means
    '''for p in range(config.eqn_config.fict_play_num-1):
        axs[0, 0].plot(x_axis, np.mean(trajec[p][0], axis=0), '--', label=p+1)
        axs[0, 1].plot(x_axis, -np.mean(trajec[p][3], axis=0), '--', label=p+1)
        axs[1, 0].plot(x_axis, np.mean(trajec[p][1], axis=0), '--', label=p+1)
        axs[1, 1].plot(x_axis, np.mean(trajec[p][2], axis=0), '--', label=p+1)'''
    '''
        axs[0, 0].plot(x_axis, np.mean(trajec[-1][0], axis=0), color='blue', label=p+1)
        axs[0, 0].set_title("Wealth mean")
        axs[0, 1].plot(x_axis, -np.mean(trajec[-1][3], axis=0), color='blue', label=p+1)
        axs[0, 1].set_title("Trading speed mean")
        axs[1, 0].plot(x_axis, np.mean(trajec[-1][1], axis=0), color='blue', label=p+1)
        axs[1, 0].set_title("Mean portfolio process")
        axs[1, 1].plot(x_axis, np.mean(trajec[-1][2], axis=0), color='blue', label=p+1)
        axs[1, 1].set_title("E[Y^0_t]")'''

    # Plotting trajectories
    p = config.eqn_config.fict_play_num
    for s in range(int(len(trajec[-1][0]) / 32)):
        axs[0].plot(x_axis, trajec[-1][0][s], label=p)
        axs[0].set_title("X trajectories")
        axs[1].plot(x_axis, trajec[-1][1][s], label=p)
        axs[1].set_title("Y trajectories")
        axs[2].plot(x_axis, trajec[-1][2][s], label=p)
        axs[2].set_title("Z trajectories")




    #plt.legend()
    plt.show()


if __name__ == '__main__':
    app.run(main)
