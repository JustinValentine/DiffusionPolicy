from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
from abc import ABC, abstractmethod

class BaseTrajectoryInterpolator(ABC):


class JointTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, joints: np.ndarray):
        assert len(times) >= 1
        assert len(joints) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(joints, np.ndarray):
            joints = np.array(joints)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._joints = joints
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            self.interp = si.interp1d(times, joints, axis=0, assume_sorted=True)

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.interp.x

    @property
    def joints(self) -> np.ndarray:
        if self.single_step:
            return self._joints
        else:
            return self.interp.y

    def trim(self, start_t: float, end_t: float) -> "JointTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_joints = self(all_times)
        return JointTrajectoryInterpolator(times=all_times, joints=all_joints)

    def drive_to_waypoint(
        self,
        joint,
        time,
        curr_time,
        max_speed=np.inf,
    ) -> "JointTrajectoryInterpolator":
        assert max_speed > 0
        time = max(time, curr_time)

        curr_joint = self(curr_time)

        joint_dist = np.abs(joint - curr_joint)
        min_duration = (joint_dist / max_speed).min()
        duration = time - curr_time
        duration = max(duration, min_duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new joint
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        joints = np.append(trimmed_interp.joints, [joint], axis=0)

        # create new interpolator
        final_interp = JointTrajectoryInterpolator(times, joints)
        return final_interp

    def schedule_waypoint(
        self, joint, time, max_speed=np.inf, curr_time=None, last_waypoint_time=None
    ) -> "JointTrajectoryInterpolator":
        assert max_speed > 0
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
        joint_dist = np.abs(end_joint - joint)
        min_duration = (joint_dist / max_speed).min()
        duration = max(duration, min_duration)
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new joint
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        joints = np.append(trimmed_interp.joints, [joint], axis=0)

        # create new interpolator
        final_interp = JointTrajectoryInterpolator(times, joints)
        return final_interp

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        joint = np.zeros((len(t), len(self.joints[0])))
        if self.single_step:
            joint[:] = self._joints[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            joint[:] = self.interp(t)

        if is_single:
            joint = joint[0]
        return joint
