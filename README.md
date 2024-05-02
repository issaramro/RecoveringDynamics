## Recovering Dynamics from Partial Measurements.

In this package, I study objects lying in the solution space of nonlinear systems of coupled ODEs when only a 1-dimensional time series from the solution space x(t) is given.
I tried combining multiple approaches with the goals:

1- Getting a qualitative understanding and making physical sense of the behavior of the governing systems from which x(t) is taken.

2- Reduce computational complexities as much as possible by using geometrical and statistical criteria on the input to be embedded.

3- Recovering sparse systems of ODEs (whose solutions do not hopefully blow up) describing the results.


<br>
Particularly, I worked with the Rossler system with the parameters (a = 0.15, b = 0.2, c = 10) and time step $dt = \frac{\pi}{100}$ where I treated the x solution as the 1-dimensional time series.
I applied the Minimum Mutual Information and the False Nearest Neighbors criteria on the time-delayed versions of x(t) in `MI_FNN.ipynb` where I recovered the embedding parameters that minimize computational complexity and dimensionality of the input ($\tau = 17$ and k = 3).

<br>
(...)
