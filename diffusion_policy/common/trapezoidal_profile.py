import numpy as np

def trapezoidal_waypoints(pos_start, pos_end, v_max, a_max):
    # Total distances for all dimensions
    total_distance = np.abs(pos_end - pos_start)
    
    # Time to accelerate to v_max for all dimensions
    t_acc = v_max / a_max
    x_acc = 0.5 * a_max * t_acc**2
    
    # Condition to check whether each trajectory has a constant velocity phase
    need_constant_velocity = 2 * x_acc <= total_distance

    # Adjust v_max for the cases where there is no constant velocity phase
    v_max_adjusted = np.where(need_constant_velocity, v_max, np.sqrt(a_max * total_distance))
    t_acc_adjusted = v_max_adjusted / a_max
    x_acc_adjusted = 0.5 * a_max * t_acc_adjusted**2

    # Constant velocity phase length for those that need it
    x_const = np.where(need_constant_velocity, total_distance - 2 * x_acc_adjusted, 0)
    t_const = np.where(need_constant_velocity, x_const / v_max_adjusted, 0)

    # Deceleration time is the same as acceleration time
    t_dec = t_acc_adjusted

    # Adjust the sign of the position based on direction
    direction = np.sign(pos_end - pos_start)
    
    # Find the maximum total time (acceleration + constant velocity + deceleration) across all profiles
    total_time = t_acc_adjusted + t_const + t_dec
    max_time = np.max(total_time)

    # Normalize each profile so that all profiles have the same total time (max_time)
    waypoint_times = np.vstack([
        np.zeros_like(total_time),                         # Start time
        t_acc_adjusted / total_time * max_time,            # End of acceleration phase
        (t_acc_adjusted + t_const) / total_time * max_time,  # End of constant velocity phase
        max_time * np.ones_like(total_time)                # End time
    ])
    
    waypoint_positions = np.vstack([
        pos_start,                                           # Start position
        pos_start + direction * x_acc_adjusted,              # End of acceleration phase
        pos_start + direction * (total_distance - x_acc_adjusted),  # End of constant velocity phase
        pos_end                                              # End position
    ])
    
    return waypoint_times, waypoint_positions
