import os
import rospy
from re import S
import time
import enum
import socket
import struct
from abc import ABC, abstractmethod
import threading
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.trajectory_interpolator import PoseTrajectoryInterpolator, JointTrajectoryInterpolator
from diffusion_policy.common.pid_controller import PIDController

from sensor_msgs.msg import JointState
from wam_msgs.msg import RTJointPos
from wam_srvs.srv import JointMove
from bhand_teleop_msgs.msg import BhandTeleop


class Command(enum.Enum):
    STOP = 0
    SERVO = 1
    SCHEDULE_WAYPOINT = 3


class Control(enum.Enum):
    POSE = 0
    JOINT = 1


class BaseInterpolationController(ABC, mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            name: str,
            shm_manager: SharedMemoryManager, 
            control_type: Control = Control.POSE,
            frequency=125, 
            launch_timeout=3,
            joints_init=None,
            joints_init_speed=1.05,
            verbose=False,
            receive_keys=None, get_max_k=128,
            **kwargs,
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time

        """
        # verify
        assert 0 < frequency <= 500
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name=name)
        if control_type == Control.POSE:
            self.interp_cls = PoseTrajectoryInterpolator
        elif control_type == Control.JOINT:
            self.interp_cls = JointTrajectoryInterpolator
        self.control_type = control_type
        
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.verbose = verbose

        self.ready_event = mp.Event()
        self.input_queue = self.create_input_queue(shm_manager)
        self.ring_buffer = self.create_ring_buffer(shm_manager, receive_keys, get_max_k)

    @abstractmethod
    def create_ring_buffer(self, shm_manager, receive_keys, get_max_k) -> SharedMemoryRingBuffer:
        pass

    @abstractmethod
    def create_input_queue(self, shm_manager) -> SharedMemoryQueue:
        pass
    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, target, duration=0.1):
        """
        duration: desired time to reach target
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        target = np.array(target)

        message = {
            'cmd': Command.SERVO.value,
            'target': target,
            'duration': duration
        }
        self.input_queue.put(message)


    def schedule_waypoint(self, target, target_time):
        assert target_time > time.time()
        target = np.array(target)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target': target,
            'target_time': target_time
        }
        self.input_queue.put(message)
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    @abstractmethod
    def run(self):
        pass

class RTDEInterpolationController(BaseInterpolationController):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,

            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        
        self.robot_ip = robot_ip
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.tcp_offset_pose = tcp_offset_pose


        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.soft_real_time = soft_real_time

        super().__init__(
            name="RTDEPositionalController",
            shm_manager=shm_manager,
            frequency=frequency,
            launch_timeout=launch_timeout,
            joints_init=joints_init,
            joints_init_speed=joints_init_speed,
            verbose=verbose,
            receive_keys=receive_keys,
            get_max_k=get_max_k
        )


    def create_ring_buffer(self, shm_manager, receive_keys, get_max_k):

        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]
        rtde_r = RTDEReceiveInterface(hostname=self.robot_ip)
        example = dict()
        for key in receive_keys:
            example[key] = np.array(getattr(rtde_r, 'get'+key)())
        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=self.frequency
        )

        self.receive_keys = receive_keys
        return ring_buffer

    def create_input_queue(self, shm_manager):

        example = {
            'cmd': Command.SERVO.value,
            'target': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )
        return input_queue


    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip
        rtde_c = RTDEControlInterface(hostname=robot_ip)
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)

        try:
            if self.verbose:
                print(f"[RTDEPositionalController] Connect to robot: {robot_ip}")

            # set parameters
            if self.tcp_offset_pose is not None:
                rtde_c.setTcp(self.tcp_offset_pose)
            if self.payload_mass is not None:
                if self.payload_cog is not None:
                    assert rtde_c.setPayload(self.payload_mass, self.payload_cog)
                else:
                    assert rtde_c.setPayload(self.payload_mass)
            
            # init pose
            if self.joints_init is not None:
                assert rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)

            # main loop
            dt = 1. / self.frequency
            curr_pose = rtde_r.getActualTCPPose()
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = self.interp_cls(
                times=[curr_t],
                values=[curr_pose]
            )
            
            iter_idx = 0
            keep_running = True
            while keep_running:
                # start control iteration
                t_start = rtde_c.initPeriod()

                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                pose_command = pose_interp(t_now)
                vel = 0.5
                acc = 0.5
                assert rtde_c.servoL(pose_command, 
                    vel, acc, # dummy, not used by ur5
                    dt, 
                    self.lookahead_time, 
                    self.gain)
                
                # update robot state
                state = dict()
                for key in self.receive_keys:
                    state[key] = np.array(getattr(rtde_r, 'get'+key)())
                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVO.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            value=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            value=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                rtde_c.waitPeriod(t_start)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            # decelerate
            rtde_c.servoStop()

            # terminate
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {robot_ip}")


class WAMInterpolationController(BaseInterpolationController):
    def __init__(self,
            shm_manager: SharedMemoryManager, 
            wam_node_prefix="/wam_master_master/follower",
            hand_node_prefix="/bhand",
            rt_control=False,
            robot_ip="192.168.1.10",
            frequency=125, 
            hand_frequency=20,
            max_speed=0.25,
            launch_timeout=3,
            joints_init=None,
            joints_init_speed=1.05,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            ):
        # TODO: fix max speed assertion
        # assert 0 < max_speed
        
        self.max_speed = max_speed
        self.jp_lock = threading.Lock()
        self.robot_ip = robot_ip

        self.wam_node_prefix = wam_node_prefix
        self.rt_control = rt_control
        self.hand_node_prefix = hand_node_prefix
        self.hand_frequency = hand_frequency
        kp = np.array([5, 5])
        self.hand_pid_controller = PIDController(
            kp=kp,
            ki=0.1*kp,
            kd=0.01*kp,
            max_velocity=np.array([2.0, 2.0]),
        )

        self.first_joint_state = False
        self.first_hand_state = False
        self.data = dict()
        self.data["hand_vel_cmd"] = np.zeros((2,), dtype=np.float64)

        super().__init__(
            name="WAMPositionalController",
            shm_manager=shm_manager,
            control_type=Control.JOINT,
            frequency=frequency,
            launch_timeout=launch_timeout,
            joints_init=joints_init,
            joints_init_speed=joints_init_speed,
            verbose=verbose,
            receive_keys=receive_keys,
            get_max_k=get_max_k
        )

    def receive_joint_state(self):
            while True:
                try:
                    data, addr = self.sock.recvfrom(1024)  # Receive data
                    unpacked_data = struct.unpack('21d', data)  # Unpack into 7 joint positions
                    positions = unpacked_data[:7]
                    velocities = unpacked_data[7:14]
                    efforts = unpacked_data[14:21]

                    # Safely write to the shared data dictionary
                    with self.jp_lock:
                        self.data['position'] = np.array(positions)  # Store the positions as a numpy array
                        self.data['velocity'] = np.array(velocities)
                        self.data['effort'] = np.array(efforts)
                        self.first_joint_state = True

                except BlockingIOError:
                    pass  # No data availableVkk

    def send_joint_positions(self, joint_positions):
        message = struct.pack('7d', *joint_positions)
        self.sock.sendto(message, (self.robot_ip, self.udp_port))

    def joint_state_callback(self, msg):
        self.data['position'] = np.array(msg.position)
        self.data['velocity'] = np.array(msg.velocity)
        self.data['effort'] = np.array(msg.effort)
        self.first_joint_state = True

    def hand_state_callback(self, msg):
        self.data['hand_position'] = np.array(msg.position)
        self.data['hand_velocity'] = np.array(msg.velocity)
        self.data['hand_effort'] = np.array(msg.effort)
        self.first_hand_state = True

    def hand_vel_cmd_callback(self, msg):
        self.data['hand_vel_cmd'] = np.array([msg.spread, msg.grasp])

    def create_ring_buffer(self, shm_manager, receive_keys, get_max_k):

        if receive_keys is None:
            receive_keys = [
                'position',
                'velocity',
                'effort',
                'hand_position',
                'hand_vel_cmd',
            ]
        example = dict()
        for key in receive_keys:
            if key in ['position', 'velocity', 'effort']:
                example[key] = np.zeros((7,), dtype=np.float64)
            elif key == 'hand_position':
                example[key] = np.zeros((8,), dtype=np.float64)
            elif key == 'hand_vel_cmd':
                example[key] = np.zeros((2,), dtype=np.float64)
            else:
                raise ValueError(f"Unhandled key {key}")

        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=self.frequency
        )

        self.receive_keys = receive_keys
        return ring_buffer

    def create_input_queue(self, shm_manager):

        example = {
            'cmd': Command.SERVO.value,
            'target': np.zeros((9,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )
        return input_queue

    # ========= main loop in process ============
    def run(self):
        rospy.init_node('wam_controller')
        # self.rt_pub = rospy.Publisher(f"{self.wam_node_prefix}/jnt_pos_cmd", RTJointPos, queue_size=10)
        self.rt_hand_pub = rospy.Publisher("/bhand_mux/bhand_vel", BhandTeleop, queue_size=10)

        # self.joint_state_sub = rospy.Subscriber(f"{self.wam_node_prefix}/joint_states", JointState, self.joint_state_callback)
        self.hand_state_sub = rospy.Subscriber(f"{self.hand_node_prefix}/joint_states", JointState, self.hand_state_callback)
        self.hand_vel_cmd_sub = rospy.Subscriber(f"{self.hand_node_prefix}/vel_cmd", BhandTeleop, self.hand_vel_cmd_callback)

        # self.joint_move = rospy.ServiceProxy(f"{self.wam_node_prefix}/joint_move", JointMove)

        hand_rel_rate = self.frequency // self.hand_frequency 

        self.udp_port = 5553
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.udp_port))
        self.sock.setblocking(False)

        self.recv_thread = threading.Thread(target=self.receive_joint_state, daemon=True)
        self.recv_thread.start()

        try:

            rate = rospy.Rate(self.frequency)

            # init pose
            if self.joints_init is not None:
                # self.joint_move(self.joints_init)
                pass

            while (not self.first_joint_state) or (not self.first_hand_state):
                rate.sleep()

            print("joint state received")

            # main loop
            dt = 1. / self.frequency
            with self.jp_lock:
                curr_pos = self.data['position']

            hand_pos = self.data['hand_position'][[0, 3]]

            curr_pos = np.append(curr_pos, hand_pos)
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pos_interp = self.interp_cls(
                times=[curr_t],
                values=[curr_pos]
            )

            prev_time = time.time()

            iter_idx = 0
            keep_running = True
            cmd_send = []
            time_sent = []
            while keep_running and not rospy.is_shutdown():
                # start control iteration

                # update robot state
                state = dict()
                for key in self.receive_keys:
                    if key not in ['position', 'velocity', 'effort']:
                        state[key] = self.data[key]
                with self.jp_lock:
                    state['position'] = self.data['position']
                    state['velocity'] = self.data['velocity']
                    state['effort'] = self.data['effort']

                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)

                # send command to robot
                t_now = time.monotonic()

                if self.rt_control:
                    pos_command = pos_interp(t_now)
                    # TODO: remove when puck 1 returned
                    pos_command[0] = 0.0

                    max_diff = self.max_speed[:7] * dt
                    pos_command[:7] = np.clip(pos_command[:7], state['position'][:7] - max_diff, state['position'][:7] + max_diff)
                    cmd_send.append(pos_command)
                    time_sent.append(time.time())

                    # self.send_joint_positions(pos_command[:7])

                    if iter_idx % hand_rel_rate == 0:
                        vel = self.hand_pid_controller.compute_velocity(
                            self.data['hand_position'][[0, 3]],
                            pos_command[7:],
                            1/self.hand_frequency
                        )
                        rt_hand_msg = BhandTeleop()
                        rt_hand_msg.spread = vel[1]
                        rt_hand_msg.grasp = vel[0]
                        # self.rt_hand_pub.publish(rt_hand_msg)



                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVO.value:

                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pos = command['target']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pos_interp = pos_interp.drive_to_waypoint(
                            value=target_pos,
                            time=t_insert,
                            curr_time=curr_time,
                            max_speed=self.max_speed,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pos, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pos_interp = pos_interp.schedule_waypoint(
                            value=target_pos,
                            time=target_time,
                            max_speed=self.max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                current_time = time.time()
                elapsed_time = current_time - prev_time
                actual_rate = 1.0 / elapsed_time
                prev_time = current_time
                
                tolerance = 0.9
                if (actual_rate / self.frequency) < tolerance:
                    print(f"Actual rate ({actual_rate}) less than desired rate ({self.frequency})")
                # regulate frequency
                rate.sleep()
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # if self.verbose:
                #     print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            # decelerate
            self.sock.close()
            self.ready_event.set()
            np.array(cmd_send).dump('cmd_send.npy')
            np.array(time_sent).dump('time_send.npy')

