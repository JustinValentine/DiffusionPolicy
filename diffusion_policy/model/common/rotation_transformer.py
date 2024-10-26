from typing import Union
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import numpy as np
import functools



class RotationTransformer:
    # used pytorch3d.transforms as reference to add more representations
    valid_reps = [
        'axis_angle',
        # 'euler_angles',
        'quaternion',
        'rotation_6d',
        # 'matrix'
    ]

    def __init__(self, 
            from_rep='axis_angle', 
            to_rep='rotation_6d', 
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(self, f'{from_rep}_to_matrix'),
                getattr(self, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convernsion=from_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(self, f'matrix_to_{to_rep}'),
                getattr(self, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convernsion=to_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])
        
        inverse_funcs = inverse_funcs[::-1]
        
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y
        
    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)
    
    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)

    @staticmethod
    def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as axis/angle to rotation matrices.

        Args:
            axis_angle: Rotations given as a vector in axis angle form,
                as a tensor of shape (..., 3), where the magnitude is
                the angle turned anticlockwise in radians around the
                vector's direction.

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        return RotationTransformer.quaternion_to_matrix(RotationTransformer.axis_angle_to_quaternion(axis_angle))

    @staticmethod
    def quaternion_to_rotation_6d(quaternions: torch.Tensor) -> torch.Tensor:
        return RotationTransformer.matrix_to_rotation_6d(RotationTransformer.quaternion_to_matrix(quaternions))

    @staticmethod
    def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:

        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    @staticmethod
    def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as axis/angle to quaternions.

        Args:
            axis_angle: Rotations given as a vector in axis angle form,
                as a tensor of shape (..., 3), where the magnitude is
                the angle turned anticlockwise in radians around the
                vector's direction.

        Returns:
            quaternions with real part first, as tensor of shape (..., 4).
        """
        angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
        half_angles = angles * 0.5
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        quaternions = torch.cat(
            [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
        )
        return quaternions

    @staticmethod
    def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)

        Returns:
            batch of rotation matrices of size (*, 3, 3)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    @staticmethod
    def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to axis/angle.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).

        Returns:
            Rotations given as a vector in axis angle form, as a tensor
                of shape (..., 3), where the magnitude is the angle
                turned anticlockwise in radians around the vector's
                direction.
        """
        return RotationTransformer.quaternion_to_axis_angle(RotationTransformer.matrix_to_quaternion(matrix))

    @staticmethod
    def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to quaternions.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).

        Returns:
            quaternions with real part first, as tensor of shape (..., 4).
        """
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        q_abs = RotationTransformer._sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )

        # we produce the desired quaternion multiplied by each of r, i, j, k
        quat_by_rijk = torch.stack(
            [
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
        # forall i; we pick the best-conditioned one (with the largest denominator)
        out = quat_candidates[
            F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
        ].reshape(batch_dim + (4,))
        return RotationTransformer.standardize_quaternion(out)

    @staticmethod
    def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert a unit quaternion to a standard form: one in which the real
        part is non negative.

        Args:
            quaternions: Quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Standardized quaternions as tensor of shape (..., 4).
        """
        return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


    @staticmethod
    def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as quaternions to axis/angle.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotations given as a vector in axis angle form, as a tensor
                of shape (..., 3), where the magnitude is the angle
                turned anticlockwise in radians around the vector's
                direction.
        """
        norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
        half_angles = torch.atan2(norms, quaternions[..., :1])
        angles = 2 * half_angles
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        return quaternions[..., 1:] / sin_half_angles_over_angles

    @staticmethod
    def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        if torch.is_grad_enabled():
            ret[positive_mask] = torch.sqrt(x[positive_mask])
        else:
            ret = torch.where(positive_mask, torch.sqrt(x), ret)
        return ret

    @staticmethod
    def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
        """
        Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
        by dropping the last row. Note that 6D representation is not unique.
        Args:
            matrix: batch of rotation matrices of size (*, 3, 3)

        Returns:
            6D rotation representation, of size (*, 6)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        batch_dim = matrix.size()[:-2]
        return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2*np.pi,2*np.pi,size=(1000,3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix
