import numpy as np
import json

import mujoco_py

from gym import error

def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                #print('bias type 0')
                sim.data.ctrl[i] = action[i]
            else:
                #print('bias type not 0')
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]

def get_joint_noise(noise_scale):
    return noise_scale*np.random.uniform(low=get_joint_limits()[:,0], high=get_joint_limits()[:,1])

def perturb_q(sim, noise_scale):
    q = get_joint_angles(sim)
    new_q = q + get_joint_noise(noise_scale)
    ctrl_set_qpos(sim, new_q)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def ctrl_set_qvel(sim, action, mode):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                #print('bias type 0')
                if mode == '4dof':
                    sim.data.set_joint_qvel('wam/base_yaw_joint',action[0])
                    sim.data.set_joint_qvel('wam/shoulder_pitch_joint',action[1])
                    sim.data.set_joint_qvel('wam/shoulder_yaw_joint',action[2])
                    sim.data.set_joint_qvel('wam/elbow_pitch_joint',action[3])
                    # sim.data.set_joint_qpos('wam/wrist_yaw_joint',0.0)
                    # sim.data.set_joint_qpos('wam/wrist_pitch_joint',0.0)
                    # sim.data.set_joint_qpos('wam/palm_yaw_joint',0.0)
                    sim.data.set_joint_qvel('wam/wrist_yaw_joint',0.0)
                    sim.data.set_joint_qvel('wam/wrist_pitch_joint',0.0)
                    sim.data.set_joint_qvel('wam/palm_yaw_joint',0.0)
                else:
                    sim.data.set_joint_qvel('wam/base_yaw_joint',action[0])
                    sim.data.set_joint_qvel('wam/shoulder_pitch_joint',action[1])
                    sim.data.set_joint_qvel('wam/shoulder_yaw_joint',action[2])
                    sim.data.set_joint_qvel('wam/elbow_pitch_joint',action[3])
                    sim.data.set_joint_qvel('wam/wrist_yaw_joint',action[4])
                    sim.data.set_joint_qvel('wam/wrist_pitch_joint',action[5])
                    sim.data.set_joint_qvel('wam/palm_yaw_joint',action[6])


def ctrl_set_qpos(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    print(action)
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                #print('bias type 0')
                sim.data.set_joint_qpos('wam/base_yaw_joint',action[0])
                sim.data.set_joint_qpos('wam/shoulder_pitch_joint',action[1])
                sim.data.set_joint_qpos('wam/shoulder_yaw_joint',action[2])
                sim.data.set_joint_qpos('wam/elbow_pitch_joint',action[3])
                sim.data.set_joint_qpos('wam/wrist_yaw_joint',action[4])
                sim.data.set_joint_qpos('wam/wrist_pitch_joint',action[5])
                sim.data.set_joint_qpos('wam/palm_yaw_joint',action[6])
            else:
                #print('bias type not 0')
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]



def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


def load_json_dict(path):
    datapath = path + '.json'
    with open(datapath, 'r') as f:
        dict = json.load(f)

    return dict


"""
WAM kinematics

from https://support.barrett.com/wiki/WAM/KinematicsJointRangesConversionFactors
"""
def get_joint_limits():
    return np.array([[-2.6, 2.6],
                     [-2.0, 2.0],
                     [-2.8, 2.8],
                     [-0.9, 3.1],
                     [-1.24, 1.24], # according to barret, should be [-4.76, 1.24]
                     [-1.6, 1.6],
                     [-3.0, 3.0],])
    
    # return np.array([[0, 0],
    #                  [0, 0],
    #                  [0, 0],
    #                  [0, 0],
    #                  [-4.76, 1.24],
    #                  [0, 0],
    #                  [0, 0],])

def get_joint_angles(data):
    return data.qpos


def get_joint_vels(data):
    return data.qvel

DH_params = [
                {'a': 0.0, 'alpha': -1.0*np.pi/2, 'd': 0.0},
                {'a': 0.0, 'alpha': np.pi/2, 'd': 0.0},          # shoulder
                {'a': 0.045, 'alpha': -1.0*np.pi/2, 'd': 0.55},
                {'a': -0.045, 'alpha': np.pi/2, 'd': 0.0},       # elbow
                {'a': 0.0, 'alpha': -1.0*np.pi/2, 'd': 0.3},
                {'a': 0.0, 'alpha': np.pi/2, 'd': 0.0},
                {'a': 0.0, 'alpha': 0.0, 'd': 0.06}
                # {'a': 0.0, 'alpha': 0.0, 'd': 0.18}
            ]

def c(theta):
    return np.cos(theta)

def s(theta):
    return np.sin(theta)



def get_DH_mat(theta, a, alpha, d):
    T = np.array([
                    [c(theta), -s(theta)*c(alpha),  s(theta)*s(alpha), a*c(theta)],
                    [s(theta),  c(theta)*c(alpha), -c(theta)*s(alpha), a*s(theta)],
                    [     0.0,           s(alpha),           c(alpha),          d],
                    [     0.0,                0.0,                0.0,        1.0]
                ])
    return T


def get_alpha(R):
    r33 = R[2,2]
    theta = np.arctan2(np.sqrt(1-r33**2),r33)
    phi = np.arctan2(R[1,2],R[0,2])
    psi = np.arctan2(R[2,1],-R[2,0])

    # theta = np.arctan2(-np.sqrt(1-r33**2),r33)
    # phi = np.arctan2(-R[1,2],-R[0,2])
    # psi = np.arctan2(-R[2,1],R[2,0])

    alpha = [phi, theta, psi]
    return alpha

def alternate_get_alpha(R):
    r33 = R[2,2]
    # theta = np.arctan2(np.sqrt(1-r33**2),r33)
    # phi = np.arctan2(R[1,2],R[0,2])
    # psi = np.arctan2(R[2,1],-R[2,0])

    theta = np.arctan2(-np.sqrt(1-r33**2),r33)
    phi = np.arctan2(-R[1,2],-R[0,2])
    psi = np.arctan2(-R[2,1],R[2,0])

    alpha = [phi, theta, psi]
    return alpha

def R_from_alpha(alpha):
    [phi, theta, psi] = alpha
    R = np.array([[c(phi)*c(theta)*c(psi)-s(phi)*s(psi), -c(phi)*c(theta)*s(psi)-s(phi)*c(psi), c(phi)*s(theta)],
                  [s(phi)*c(theta)*c(psi)+c(phi)*s(psi), -s(phi)*c(theta)*s(psi)+c(phi)*c(psi), s(phi)*s(theta)],
                  [-s(theta)*c(psi),                           s(theta)*s(psi),                       c(theta)]])
    return R

def get_B(alpha):
    #[phi, theta, psi] = alpha
    [psi, theta, phi] = alpha # phi and psi swapped in textbook
    
    B = np.array([
                    [c(psi)*s(theta), -s(psi), 0],
                    [s(psi)*s(theta),  c(psi), 0],
                    [c(theta),         0,      1]
                 ])

    return B

def get_analytic_J(T, J):
    R = T[:3,:3]
    alpha = get_alpha(R)
    B = get_B(alpha)
    Binv = np.linalg.inv(B)
    I = np.eye(3)
    Z = np.zeros((3,3))
    block = np.block([[I,    Z],
                      [Z, Binv]])

    #print('block', block)
    J_a = np.matmul(block,J)
    #print('J_a', J_a)
    return J_a    

def alternate_get_euler_angles(q, n_joints):
    '''
    Returns Euler angles alpha = [phi, theta, psi]
    See Spong robotics book
    phi is rotation about original z
    theta is rotation about new y
    psi is rotation about new z
    '''
    T = get_kinematics_mat(q, n_joints)
    R = T[:3,:3]
    alpha = alternate_get_alpha(R)
    return alpha

def get_euler_angles(q, n_joints):
    '''
    Returns Euler angles alpha = [phi, theta, psi]
    See Spong robotics book
    phi is rotation about original z
    theta is rotation about new y
    psi is rotation about new z
    '''
    T = get_kinematics_mat(q, n_joints)
    R = T[:3,:3]
    alpha = get_alpha(R)
    return alpha

def WAM_fwd_kinematics_mat(q, n_joints):
    T_7to0 = np.eye(4)
    for i in range(n_joints-1,-1,-1):  # loops backwards from 6, 5, 4, ... , 0
        #print(i)
        T_7to0 = np.matmul(get_DH_mat(q[i], DH_params[i]['a'], DH_params[i]['alpha'], DH_params[i]['d']),T_7to0)
    return T_7to0

def WAM_fwd_kinematics(q, n_joints):
    T_7to0 = np.eye(4)
    for i in range(n_joints-1,-1,-1):  # loops backwards from 6, 5, 4, ... , 0
        T_7to0 = np.matmul(get_DH_mat(q[i], DH_params[i]['a'], DH_params[i]['alpha'], DH_params[i]['d']),T_7to0)
    
    [Px, Py, Pz] = T_7to0[0:3,3]
    return np.array([Px, Py, Pz]) #+ np.array([0.0, 0.0, 0.346])
    #return T_7to0

def WAM_fwd_kinematics_orientation(q, n_joints):
    T_7to0 = np.eye(4)
    for i in range(n_joints-1,-1,-1):  # loops backwards from 6, 5, 4, ... , 0
        T_7to0 = np.matmul(get_DH_mat(q[i], DH_params[i]['a'], DH_params[i]['alpha'], DH_params[i]['d']),T_7to0)
    
    [zx, zy, zz] = T_7to0[0:3,2]
    return np.array([zx, zy, zz])

def get_kinematics_mat(q, n_joints):
    T_7to0 = np.eye(4)
    for i in range(n_joints-1,-1,-1):  # loops backwards from 6, 5, 4, ... , 0
        #print(i)
        T_7to0 = np.matmul(get_DH_mat(q[i], DH_params[i]['a'], DH_params[i]['alpha'], DH_params[i]['d']),T_7to0)
    return T_7to0

def get_J(q, n_joints):
    # Calculated following formula in Spong robot modeling and control
    J = np.eye(6)
    z_prev = np.array([0, 0, 1]).T # unit z vector
    o_prev = np.array([0, 0, 0]).T # origin
    o_n = get_kinematics_mat(q, n_joints)[0:3,3] # coordinates of ee
    #print('o_n', o_n)
    T_ito0 = np.eye(4)
    J = None
    for i in range(n_joints):
        #print(i)
        T_ito0 = np.matmul(T_ito0, get_DH_mat(q[i], DH_params[i]['a'], DH_params[i]['alpha'], DH_params[i]['d']))
        J_i = np.concatenate((np.cross(z_prev,(o_n-o_prev)),z_prev),axis=0)
        J_i = np.expand_dims(J_i,1) # turn into col vector
        z_prev = T_ito0[:3,2] # first three elements of third col
        o_prev = T_ito0[:3,3] # fouth col
        #print('J_i',J_i)
        #print(J)
        if J is not None:
            J = np.concatenate((J,J_i),axis=1)
        else:
            J = J_i
        #print('J after', J)

        
    return [J, T_ito0]
    

def get_spherical_coords(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan(np.sqrt(x**2+y**2)/z)
    phi = np.arctan(y/x) - np.pi # assumes x<0, y<0 TODO: use atan2
    return r, theta, phi

def get_J_s_x(x, y, z): 
    # Jacobian describing how spherical coordinate change as function of Cartesian coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    
    drdx = x/r
    drdy = y/r
    drdz = z/r

    dthetadx = 1/(1 + (x**2+y**2)/z**2) * x/(z*np.sqrt(x**2+y**2))
    dthetady = 1/(1 + (x**2+y**2)/z**2) * y/(z*np.sqrt(x**2+y**2))
    dthetadz = -1/r**2

    dphidx = -y/(x**2+y**2)
    dphidy = 1/(x+y**2/x)
    dphidz = 0

    J_s_x = np.array([
                        [drdx,     drdy,     drdz    ],
                        [dthetadx, dthetady, dthetadz],
                        [dphidx,   dphidy,   dphidz  ]
                    ])
    
    return J_s_x


class RingBuffer():
    """
    Simple RingBuffer object to hold values to average (useful for, e.g.: filtering D component in PID control)

    Note that the buffer object is a 2D numpy array, where each row corresponds to
    individual entries into the buffer

    Args:
        dim (int): Size of entries being added. This is, e.g.: the size of a state vector that is to be stored
        length (int): Size of the ring buffer
    """

    def __init__(self, dim, length):
        # Store input args
        self.dim = dim
        self.length = length

        # Variable so that initial average values are accurate
        self._size = 0

        # Save pointer to end of buffer
        self.ptr = self.length - 1

        # Construct ring buffer
        self.buf = np.zeros((length, dim))

    def push(self, value):
        """
        Pushes a new value into the buffer

        Args:
            value (int or float or array): Value(s) to push into the array (taken as a single new element)
        """
        # Increment pointer, then add value (also increment size if necessary)
        self.ptr = (self.ptr + 1) % self.length
        self.buf[self.ptr] = np.array(value)
        if self._size < self.length:
            self._size += 1

    def clear(self):
        """
        Clears buffer and reset pointer
        """
        self.buf = np.zeros((self.length, self.dim))
        self.ptr = self.length - 1
        self._size = 0

    @property
    def current(self):
        """
        Gets the most recent value pushed to the buffer

        Returns:
            float or np.array: Most recent value in buffer
        """
        return self.buf[self.ptr]

    @property
    def average(self):
        """
        Gets the average of components in buffer

        Returns:
            float or np.array: Averaged value of all elements in buffer
        """
        return np.mean(self.buf[: self._size], axis=0)