import numpy as np

def trapezoidal_waypoints(pos_start, pos_end, v_max, a_max, time_step= 0.1):
   
    num_joints = len(pos_start)
    delta_pos = pos_end - pos_start
    direction = np.sign(delta_pos)

    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2

    d_total = np.abs(delta_pos)
    d_flat = d_total - 2 * d_acc

    for i in range(num_joints):
        if d_flat[i] < 0:
            t_acc[i] = np.sqrt(d_total[i] / a_max[i])
            d_acc[i] = 0.5 * a_max[i] * t_acc[i]**2
            d_flat[i] = 0
        t_flat = d_flat / v_max
        
    t_total = 2 * t_acc + t_flat
    t = np.arange(0, t_total.max(), time_step)
    
    pos = np.zeros((len(t), num_joints))

    for i in range(num_joints):
        for j, ti in enumerate(t):
            if ti < t_acc[i]:
                pos[j, i] = pos_start[i] + direction[i] * 0.5 * a_max[i] * ti**2
            elif t_acc[i] <= ti < (t_acc[i] + t_flat[i]):
                pos[j, i] = pos_start[i] + direction[i] * (d_acc[i] + v_max[i] * (ti - t_acc[i]))
            elif (t_acc[i] + t_flat[i]) <= ti < t_total[i]:
                pos[j, i] = pos_end[i] - direction[i] * 0.5 * a_max[i] * (t_total[i] - ti)**2
            else:
                pos[j, i] = pos_end[i]

    
    return t, pos
