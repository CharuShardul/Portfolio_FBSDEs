# Long Term Dynamic Portfolio Optimization using Infinite Horizon McKean-Vlasov FBSDEs

This project implements a **global direct solver** for Forward-Backward Stochastic Differential Equations (FBSDEs) using deep neural networks (TensorFlow/Keras). It is designed for long-term dynamic portfolio optimization problems, as described in the [PhD thesis of Charu Shardul](https://theses.hal.science/tel-04627360v1).

---

## Mathematical Formulation

The system is based on a coupled FBSDE of the form:

\[
\begin{cases}
dX_t = b(t, X_t, Y_t, Z_t)dt + \sigma(t, X_t, Y_t, Z_t)dW_t \\
dY_t = -f(t, X_t, Y_t, Z_t)dt + Z_t dW_t \\
X_0 = x_0, \quad Y_T = g(X_T)
\end{cases}
\]

where:
- \( X_t \) is the forward process (e.g., wealth, portfolio state)
- \( Y_t \) is the backward process (e.g., value function, adjoint)
- \( Z_t \) is the control/gradient process
- \( W_t \) is a Brownian motion
- \( b, \sigma, f, g \) are problem-specific functions

For the **McKean-Vlasov** (mean-field) case, coefficients may depend on the law of \( X_t \).

### Example: Portfolio Optimization FBSDE

The specific FBSDE solved in this code is:

\[
\begin{align*}
dX_t^0 &= \left[ a_V(S_t) X_t^0 + a_d(S_t) X_t^1 \right] dt + \left[ b_V(S_t) X_t^0 + b_d(S_t) X_t^1 \right] dW_t \\
dX_t^1 &= -Y_t^1 dt \\
dY_t^0 &= -f_0(t, X_t, Y_t, Z_t, S_t, \zeta) dt + Z_t^0 dW_t \\
dY_t^1 &= -f_1(t, X_t, Y_t, Z_t, S_t, \zeta) dt + Z_t^1 dW_t
\end{align*}
\]

where the drivers \( f_0, f_1 \) are given by:
\[
\begin{align*}
f_0 &= Y_t^0 a_V(S_t) + Z_t^0 b_V(S_t) - \rho Y_t^0 + \alpha \, \text{ldx}(t, X_t^0, \zeta) \\
f_1 &= Y_t^0 a_d(S_t) + Z_t^0 b_d(S_t) - \rho Y_t^1
\end{align*}
\]

The function `ldx` encodes the law-dependent (mean-field) term.

---

## Main Parameters (`FBSDE_config_1d.json`)

The main configuration file is [`configs/FBSDE_config_1d.json`](configs/FBSDE_config_1d.json). **Key parameters:**

### Equation Parameters (`eqn_config`)
| Parameter         | Description                                      | Example Value         |
|-------------------|--------------------------------------------------|----------------------|
| `eqn_name`        | Equation class name                              | `"FBSDE"`            |
| `r`               | Interest rate                                    | `0.026`              |
| `rho`             | Discount factor                                  | `0.2`                |
| `gamma`           | Risk aversion parameters                         | `[400.0, 8.317]`     |
| `X_dim`           | State dimension                                  | `2`                  |
| `Y_dim`           | Backward variable dimension                      | `2`                  |
| `W_dim`           | Brownian motion dimension                        | `1`                  |
| `X_init`          | Initial state                                    | `[1.0, 8.317]`       |
| `lamb`            | Smoothing parameter for law term                 | `10.0`               |
| `t_grid_size`     | Number of time steps per unit time               | `6`                  |
| `total_time`      | Total time horizon                               | `6.0`                |
| `fict_play_num`   | Fictitious play iterations                       | `2`                  |
| `nu_support`      | Support for target distribution (law term)       | `[0.92, ..., 1.08]`  |

### Neural Network Parameters (`net_config`)
| Parameter         | Description                                      | Example Value         |
|-------------------|--------------------------------------------------|----------------------|
| `y_init_range`    | Initial range for \( Y_0 \)                      | `[0.0, 0.1]`         |
| `lr_values`       | Learning rate schedule values                    | `[1e-2, ..., 5e-6]`  |
| `lr_boundaries`   | Iteration boundaries for learning rate changes   | `[100, 500, ...]`    |
| `num_iterations`  | Number of training iterations                    | `2000`               |
| `batch_size`      | Training batch size                              | `32`                 |
| `valid_size`      | Validation batch size                            | `256`                |
| `logging_frequency`| Logging frequency (iterations)                  | `100`                |
| `dtype`           | Data type                                        | `"float32"`          |
| `verbose`         | Verbosity flag                                   | `true`               |

---

## Usage

1. **Edit configuration** in `configs/FBSDE_config_1d.json` as needed.
2. **Run the main script:**
   ```bash
   python main.py --config_path configs/FBSDE_config_1d.json --stock_data Data_files/b_and_sig.json --exp_name my_experiment
   ```
3. **Results** (trajectories, means, plots) are saved in the `logs/` and `Numerical_experiments/` directories.

---

## References

- Charu Shardul, [PhD Thesis](https://theses.hal.science/tel-04627360v1)
- Han, J., & Long, J. (2022). Convergence of Deep BSDE method for Coupled FBSDE.

---

## License

This code is for academic use. Please cite the thesis if you use it in your research.
