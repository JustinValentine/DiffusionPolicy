import numpy as np
from diffusion_policy.common.trapezoidal_profile import trapezoidal_waypoints
import matplotlib.pyplot as plt

def test_trapezoidal_profile():
    times, positions = trapezoidal_waypoints(np.array([0, 0, 0]), np.array([1, 2, 3]), 1, 1)
    for i in range(3):
        plt.plot(times[:, i], positions[:, i])
        plt.show()
