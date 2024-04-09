from ml_model import nn
from numerical_model import lorenz96

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Time step for the ML model; use the same for the numerical integration
dt = 0.05
n_steps = 100

# Generate a random state
x0 = np.random.randn(40)

x = x0
x_ml = np.zeros((n_steps, 40))
x_ml[0] = x0
for i in range(1, n_steps):
    x = nn._smodel.predict(x.reshape((1, 40, 1)))[0, :, 0]
    x_ml[i] = x

x_phys = solve_ivp(lorenz96, [0, n_steps*dt], x0, t_eval=np.arange(0.0, n_steps*dt, dt)).y.T

# Plot the RMSE between the physical and ML forecast
plt.plot(np.sqrt(((x_ml - x_phys)**2).mean(axis=1)))

plt.savefig("rmse.pdf")
