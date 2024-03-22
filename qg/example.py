from ml_model import model
from physical_model import drymodel

import numpy as np
import matplotlib.pyplot as plt

# The step of each of the models is 1.0, corresponding to ~6 hours

n_transient = 100
n_steps = 40

psi0 = np.random.randn(96, 192, 2)

psi = psi0
for i in range(n_transient):
    psi = drymodel(psi)

# Integrate forward numerical and ML model
psi0 = psi

psi = psi0
psis_phys = [psi0]
for i in range(n_steps):
    psi = drymodel(psi)
    psis_phys.append(psi)

psi = psi0
psis_ml = [psi0]
for i in range(n_steps):
    psi = model.predict(psi.transpose((1, 0, 2))[np.newaxis, :])[0, :, :, :].transpose((1, 0, 2))
    psis_ml.append(psi)

# Plot the RMSE between the physical and ML forecast
plt.plot(np.sqrt(np.mean((np.array(psis_ml) - np.array(psis_phys))**2, axis=(1, 2, 3))))