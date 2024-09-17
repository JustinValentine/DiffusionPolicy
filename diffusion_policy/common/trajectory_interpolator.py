from abc import ABC, abstractmethod
from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st

def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()

def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist

class BaseTrajectoryInterpolator(ABC):
    def __init__(self, times: Union[list, np.ndarray], values: Union[list, np.ndarray]):
        assert len(times) >= 1
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._values = values
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            self.setup_interpolator(times, values)
        pass


    @abstractmethod
    def setup_interpolator(self, times, values):
        pass

    @abstractmethod
    def interpolate(self, times) -> np.ndarray:
        pass

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        if self.single_step:
            result = np.tile(self._values[0], (len(t), 1))
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            result = self.interpolate(t)

        if is_single:
            result = result[0]
        return result

    @property
    @abstractmethod
    def times(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def values(self) -> np.ndarray:
        pass

    def trim(self, 
            start_t: float, end_t: float
            ) -> "BaseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_values = self(all_times)
        return self.__class__(times=all_times, values=all_values)


class PoseTrajectoryInterpolator(BaseTrajectoryInterpolator):
    
    def setup_interpolator(self, times, values):
        pos = values[:,:3]
        rot = st.Rotation.from_rotvec(values[:,3:])

        self.pos_interp = si.interp1d(times, pos, 
            axis=0, assume_sorted=True)
        self.rot_interp = st.Slerp(times, rot)
    
    @property
    def values(self) -> np.ndarray:
        if self.single_step:
            return self._values
        else:
            n = len(self.times)
            poses = np.zeros((n, 6))
            poses[:,:3] = self.pos_interp.y
            poses[:,3:] = self.rot_interp(self.times).as_rotvec()
            return poses

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pos_interp.x
    
    def interpolate(self, times):
        pose = np.zeros((len(times), 6))
        pose[:,:3] = self.pos_interp(times)
        pose[:,3:] = self.rot_interp(times).as_rotvec()
        return pose

    def drive_to_waypoint(self, 
            value, time, curr_time,
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        time = max(time, curr_time)
        
        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, value)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new pose
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.values, [value], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def schedule_waypoint(self,
            value, time, 
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)
        
        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(value, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.values, [value], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp




class JointTrajectoryInterpolator(BaseTrajectoryInterpolator):
    def setup_interpolator(self, times: np.ndarray, values: np.ndarray):
        self.interp = si.interp1d(times, values, axis=0, assume_sorted=True)


    @property
    def values(self) -> np.ndarray:
        if self.single_step:
            return self._values
        else:
            return self.interp.y

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.interp.x

    def interpolate(self, times):
        return self.interp(times)

    def drive_to_waypoint(
        self,
        value,
        time,
        curr_time,
        max_speed=np.inf,
    ) -> "JointTrajectoryInterpolator":
        assert max_speed > 0
        time = max(time, curr_time)

        curr_joint = self(curr_time)

        joint_dist = np.abs(value - curr_joint)
        min_duration = (joint_dist / max_speed).min()
        duration = time - curr_time
        duration = max(duration, min_duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new joint
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        joints = np.append(trimmed_interp.values, [value], axis=0)

        # create new interpolator
        final_interp = JointTrajectoryInterpolator(times, joints)
        return final_interp

    def schedule_waypoint(
        self, value, time, max_speed=np.inf, curr_time=None, last_waypoint_time=None
    ) -> "JointTrajectoryInterpolator":
        # assert max_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)

        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_joint = trimmed_interp(end_time)
        joint_dist = np.abs(end_joint - value)
        min_duration = (joint_dist / max_speed).max()
        duration = max(duration, min_duration)
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new joint
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        joints = np.append(trimmed_interp.values, [value], axis=0)

        # create new interpolator
        final_interp = JointTrajectoryInterpolator(times, joints)
        return final_interp

