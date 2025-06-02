from Onedim_NN import FBSDEsolver

from absl import app
from absl import flags
from absl import logging as absl_logging

flags.DEFINE_string('config_path', r'.\configs\FBSDE_config.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', 'test',
                    """The name of numerical experiments, prefix for logging""")
FLAGS = flags.FLAGS
FLAGS.log_dir = './logs'  # directory where to write event logs and output array
