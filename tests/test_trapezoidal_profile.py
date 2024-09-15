import numpy as np
from diffusion_policy.common.trapezoidal_profile import trapezoidal_waypoints
import matplotlib.pyplot as plt

def test_trapezoidal_profile():
    cur_pos = np.array([-0.00650116, -2.03248382,  0.03744754,  2.55714597, -0.088876,    0.64348122,
      0.07058572,  0.,         3.1565790216853884])
    goal_pos =  np.array([-3.83049790e-05,  8.52910535e-02, -6.12492730e-02,  2.77529342e+00,
      3.49494592e-02,  1.31304435e-01,  4.03060575e-02,  0.00000000e+00,
      0.00000000e+00])
    v_max = np.array([0.5]*9)
    a_max = np.array([0.5]*9)
    times, positions = trapezoidal_waypoints(cur_pos, goal_pos, v_max, a_max, time_step=0.1)
    for i in range(9):
        print(f'Joint {i} positions: {positions[:, i]}')
        plt.plot(times, positions[:, i])
        plt.title(f'Joint {i}')
        plt.show()

