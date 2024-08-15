from tqdm import tqdm
import numpy as np
from diffusion_policy.common.joint_trajectory_interpolator import (
    JointTrajectoryInterpolator,
)
from numpy.testing import assert_array_equal


def test_basic_joint_trajectory_interpolator():
    t = np.linspace(0, 2, 5)
    joints = np.array(
        [
            [0],
            [1],
            [2],
        ]
    )

    interp = JointTrajectoryInterpolator([0, 1, 2], joints)
    actual = interp(t)
    expected = np.linspace([0], [2], 5)
    assert_array_equal(actual, expected)

    t = np.linspace(0, 2, 5)
    joints = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
        ]
    )

    interp = JointTrajectoryInterpolator([0, 1, 2], joints)
    actual = interp(t)
    expected = np.linspace([0, 0], [2, 2], 5)
    assert_array_equal(actual, expected)


def test_joint_trajectory_interpolator():
    t = np.linspace(-1, 5, 100)
    interp = JointTrajectoryInterpolator([0, 1, 3], np.zeros((3, 6)))
    times = interp.times
    joints = interp.joints
    assert (times == [0, 1, 3]).all()
    assert (joints == np.zeros((3, 6))).all()

    trimmed_interp = interp.trim(-1, 4)
    assert len(trimmed_interp.times) == 5
    trimmed_interp(t)

    trimmed_interp = interp.trim(0.5, 3.5)
    assert len(trimmed_interp.times) == 4
    trimmed_interp(t)

    trimmed_interp = interp.trim(0.5, 2.5)
    assert len(trimmed_interp.times) == 3
    trimmed_interp(t)

    trimmed_interp = interp.trim(0.5, 1.5)
    assert len(trimmed_interp.times) == 3
    trimmed_interp(t)

    trimmed_interp = interp.trim(1.2, 1.5)
    assert len(trimmed_interp.times) == 2
    trimmed_interp(t)

    trimmed_interp = interp.trim(1.3, 1.3)
    assert len(trimmed_interp.times) == 1
    trimmed_interp(t)


def test_schedule_waypoint():
    # fuzz testing
    for i in tqdm(range(10000)):
        rng = np.random.default_rng(i)
        n_waypoints = rng.integers(1, 5)
        waypoint_times = np.sort(rng.uniform(0, 1, size=n_waypoints))
        last_waypoint_time = waypoint_times[-1]
        insert_time = rng.uniform(-0.1, 1.1)
        curr_time = rng.uniform(-0.1, 1.1)
        max_speed = rng.poisson(3) + 1e-3
        waypoint_joints = rng.normal(0, 3, size=(n_waypoints, 6))
        new_joint = rng.normal(0, 3, size=6)

        if rng.random() < 0.1:
            last_waypoint_time = None
            if rng.random() < 0.1:
                curr_time = None

        interp = JointTrajectoryInterpolator(
            times=waypoint_times, joints=waypoint_joints
        )
        new_interp = interp.schedule_waypoint(
            joint=new_joint,
            time=insert_time,
            max_speed=max_speed,
            curr_time=curr_time,
            last_waypoint_time=last_waypoint_time,
        )


def test_drive_to_waypoint():
    # fuzz testing
    for i in tqdm(range(10000)):
        rng = np.random.default_rng(i)
        n_waypoints = rng.integers(1, 5)
        waypoint_times = np.sort(rng.uniform(0, 1, size=n_waypoints))
        insert_time = rng.uniform(-0.1, 1.1)
        curr_time = rng.uniform(-0.1, 1.1)
        max_speed = rng.poisson(3) + 1e-3
        waypoint_joints = rng.normal(0, 3, size=(n_waypoints, 6))
        new_joint = rng.normal(0, 3, size=6)

        interp = JointTrajectoryInterpolator(
            times=waypoint_times, joints=waypoint_joints
        )
        new_interp = interp.drive_to_waypoint(
            joint=new_joint,
            time=insert_time,
            curr_time=curr_time,
            max_speed=max_speed,
        )


if __name__ == "__main__":
    test_drive_to_waypoint()
