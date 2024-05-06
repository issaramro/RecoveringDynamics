## Recovering Dynamics from Partial Measurements

This repository is an attempt to recover the dynamics of objects lying in the solution space of nonlinear systems of coupled ODEs when only a 1-dimensional time series $ x(t) $ is available, mainly following the paper by Bakarji et. al. https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2023.0422.


### Objectives:

1. Getting a qualitative understanding and making physical sense of the behavior of the governing systems from which x(t) is taken.
   
2. Reducing computational complexities as much as possible by using geometrical and statistical criteria on the data to be embedded.

3. Recovering sparse systems of ODEs (whose solutions do not hopefully blow up) describing the results.

### Approach:

This work focuses mainly on the Rossler system with the parameters $ (a = 0.15, b = 0.2, c = 10) $ and a time step $ dt = \frac{\pi}{100} $, treating the $x(t)$ solution as a 1-dimensional time series.

### Methods:

1. **Mutual Information and False Nearest Neighbors**: These criteria are applied on time-delayed versions of $ x(t) $ in `MI_FNN_rossler.ipynb`, utilizing functions defined in `functions.py`. The recovered embedding parameters $(\tau = 17, k = 3) $ are then used to identify the 3 embedding variables $(v_1, v_2, v_3) $.

2. **Polynomial Approach**: An attempt to relate the original Rossler variables to the embedding variables through a second-order polynomial is explored in `poly_approach.ipynb`.

3. **Autoencoder (AE) Implementation**:
    - `ae.py` contains an AE object used in different scenarios:
        - **Supervised Case**: Tackled in `supervised_ae.ipynb`.
        - **Known System Case**: Tackled in `ae_known_equ.ipynb`.
        - **Unknown System (Reconstruction and 1st comp losses only)**: Tackled in `unsupervised_ae.ipynb`.
        - **Unknown System (SINDy Losses applied)**: Tackled in `SINDyAutoencoder.ipynb`.
    
4. A modified version of some `latent model discovery` approach codes is provided in the folder `lmd modified`, where the encoder, decoder, hypothesis system, and integrator are defined separately and fed to the AE object in `models.py`.
   
    This approach is tested on embedding variables with all SINDy Autoencoder losses applied in two cases:
    - **Known System**: Explored in `rossler_known_equ.ipynb`.
    - **Unknown System**: Explored in `rossler_test.ipynb`.
