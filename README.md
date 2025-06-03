# Long Term Dynamic Portfolio Optimization using Infinite Horizon McKean-Vlasov FBSDEs

This project implements a test-case for the **global direct solver** approach for a McKean-Vlasov Forward-Backward Stochastic Differential Equation (FBSDE) using deep neural networks (TensorFlow/Keras). It is designed for the long-term dynamic portfolio optimization problem, as described in the [PhD thesis of Charu Shardul](https://theses.hal.science/tel-04627360v1).

---

## General Mathematical Formulation

The system is based on a coupled FBSDE of the form:

$$
\begin{aligned}
dX_t &= b(t, X_t, Y_t, Z_t)dt + \sigma(t, X_t, Y_t, Z_t)dW_t \\
dY_t &= -f(t, X_t, Y_t, Z_t)dt + Z_t dW_t \\
X_0 &= x_0, \quad Y_T = g(X_T)
\end{aligned}
$$

where:
- $X_t$ is the forward process (e.g., wealth, portfolio state)
- $(Y_t, Z_t)$ are the backward processes (e.g., adjoint processes for an optimization problem)
- $W_t$ is a Brownian motion
- $(b, \sigma, f, g)$ are problem-specific functions

For the **McKean-Vlasov** (mean-field) case, the functions $(b, \sigma, f, g)$ may depend on the law of $ X_t $ in general.

## Portfolio Optimization FBSDE

Consider a portfolio consisting of $N$ risky assets and a risk-free asset which satisfy the following SDEs and ODE respectively,
$$
\begin{aligned}
 d S^i_t &= S^i_t\left(b_t^i dt + \sum_{j=1}^m\sigma_t^{i,j} dW_t^j  \right),\ \text{for } i\in\{1, 2, \dots, N\},\\
d S^0_t &= S^0_t r_t dt.
\end{aligned}
$$

For a given initial endowment $U_{t=0} = U_0$ and the portfolio given by $\{U_t - \sum_{i=1}^N X^i_t S^i_t, X^1_t, \dots, X^N_t \}$, define the relative wealth process $X^0= \frac{U_t}{\sum_{i=0}^N S^i_t\gamma^i}$ where the portfolio given by $\gamma^i$ is called the reference portfolio. 

Our portfolio optimization criterion is based on a certain 'distance' from a target distribution parametrized by $(p_k)_{k\in \mathbb R}$. We want to minimize a cost functional of the form,

$$
J^\lambda(q) = \mathbb E\left[\int_0^\infty e^{-\rho t}\left(g(X_t, q_t) + \int_{-\infty}^\infty  \Big[\mathbb E[k-X^0_t]^\lambda_+ - p_k\Big]^\lambda_+ d\nu(k)\right) dt \right].
$$

The corresponding McKean-Vlasov FBSDE solved in this code is:

$$
\begin{aligned}
dX_t^0 &= \left( \alpha_V(S_t) X_t^0 + \alpha_d(S_t) X_t^1 \right) dt + \left( \beta_V(S_t) X_t^0 + \beta_d(S_t) X_t^1 \right) dW_t \\
dX_t^1 &= -Y_t^1 dt \\
dY_t^0 &= -f_0(t, X_t, Y_t, Z_t, S_t, \zeta_t) dt + Z_t^0 dW_t \\
dY_t^1 &= -f_1(t, X_t, Y_t, Z_t, S_t, \zeta_t) dt + Z_t^1 dW_t
\end{aligned}
$$

The drivers \( f_0, f_1 \) are given by:

$$
\begin{aligned}
f_0 &= Y_t^0 \alpha_V(S_t) + Z_t^0 \beta_V(S_t) - \rho Y_t^0 + \alpha \, \zeta_t \\
f_1 &= Y_t^0 \alpha_d(S_t) + Z_t^0 \beta_d(S_t) - \rho Y_t^1
\end{aligned}
$$

The process $(\zeta_t)_{t\in [0, T]}$ encodes the law-dependence through Lion's derivative (see equation 3.4.3 in the thesis).

### Variables' description (`Global_direct_solver.py`)
| Parameter         | Description                                      | Example Value         |
|-------------------|--------------------------------------------------|----------------------|
| `x0`               | Relative wealth process                              | starting value `1.0`            |
| `x1`               | Number of units of risky asset                       | starting value `8.3`              |
| `y0`             | Adjoint variable                            | Target terminal value `0.0`                |
| `y1`            |  Negative trading speed of the risky asset                   |  Target terminal value `0.0`|
| `zeta`          | Lion's derivative of the distance between distributions| eg. `0.36` |
| `alpha`         | Penalty on the distance between distributions       |  eg. `2.0`         |


---

## Main Parameters (`FBSDE_config_1d.json`)

The main configuration file is [`configs/FBSDE_config_1d.json`](configs/FBSDE_config_1d.json). Below are key parameters for easy customization:

### Equation Parameters (`eqn_config`)
| Parameter         | Description                                      | Example Value         |
|-------------------|--------------------------------------------------|----------------------|
| `eqn_name`        | Equation class name                              | `"FBSDE"`            |
| `r`               | Interest rate                                    | `0.026`              |
| `rho`             | Discount factor                                  | `0.2`                |
| `gamma`           | Initial portfolio weights                        | `[400.0, 8.317]`    |
| `X_dim`           | State dimension                                  | `2`                  |
| `Y_dim`           | Backward variable dimension                      | `2`                  |
| `W_dim`           | Brownian motion dimension                        | `1`                  |
| `X_init`          | Initial state                                    | `[1.0, 8.317]`       |
| `lamb`            | Smoothing parameter for law term                 | `10.0`               |
| `t_grid_size`     | Number of time steps per unit time               | `6`                  |
| `total_time`      | Total time horizon                               | `6.0`                |
| `nu_support`      | Support for target distribution                  | `[0.92, ..., 1.08]`  |

### Neural Network Parameters (`net_config`)
| Parameter         | Description                                      | Example Value         |
|-------------------|--------------------------------------------------|----------------------|
| `y_init_range`    | Initial range for \( Y_0 \)                      | `[0.0, 0.1]`         |
| `num_iterations`  | Number of training iterations                    | `2000`               |
| `batch_size`      | Training batch size                              | `32`                 |
| `valid_size`      | Validation batch size                            | `256`                |
| `logging_frequency`| Logging frequency (iterations)                  | `100`                |

---

## Usage

Follow these steps to run the code:

1. **Edit the configuration** in `configs/FBSDE_config_1d.json` as needed.
2. **Run the main script:**

   ```bash
   python main.py --config_path configs/FBSDE_config_1d.json --stock_data Data_files/b_and_sig.json --exp_name my_experiment

