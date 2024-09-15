import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, max_velocity):
        """
        Initialize the PID controller.
        :param kp: Proportional gain.
        :param ki: Integral gain.
        :param kd: Derivative gain.
        :param max_velocity: Maximum velocity allowed for the joints.
        """
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.max_velocity = max_velocity  # Velocity limit for each joint

        # Internal state
        self.previous_error = None  # For storing the previous error (for derivative term)
        self.integral_error = 0.0  # For accumulating the integral of the error

    def compute_velocity(self, current_position, desired_position, dt):
        """
        Compute the velocity command using the PID controller.
        :param current_position: Current position of the joints.
        :param desired_position: Desired position of the joints.
        :param dt: Time step.
        :return: Velocity command for each joint.
        """
        error = desired_position - current_position  # Position error

        # Proportional term
        proportional = self.kp * error

        # Integral term: accumulate error over time
        self.integral_error += error * dt
        self.inegral_error = np.clip(self.integral_error, -0.7*self.max_velocity, 0.7*self.max_velocity)
        integral = self.ki * self.integral_error

        # Derivative term: rate of change of error
        derivative = 0.0
        if self.previous_error is not None:
            derivative = self.kd * (error - self.previous_error) / dt
        self.previous_error = error

        # PID control output
        velocity_command = proportional + integral + derivative

        # Clip the velocity to the maximum allowed range
        velocity_command = np.clip(velocity_command, -self.max_velocity, self.max_velocity)

        return velocity_command

    def reset(self):
        """
        Reset the PID controller state (e.g., when switching to a new target).
        """
        self.previous_error = None
        self.integral_error = 0.0

