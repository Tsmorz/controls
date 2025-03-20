"""Add a doc string to my files."""

import numpy as np
import scipy
from loguru import logger
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from sympy import Matrix

from config.definitions import EULER_ORDER


def skew_matrix(vector: np.ndarray) -> np.ndarray:
    """Calculate the skew symmetric matrix from a given vector.

    :param vector: A 3D vector represented as a numpy array.
    :return: The skew symmetric matrix of the given vector.
    """
    dim = len(np.shape(vector))
    if dim == 2:
        vector = np.reshape(vector, (3,))
    if len(vector) != 3:
        raise ValueError("Input vector must have a dimension of 3 or less.")

    sk = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )
    return sk


def roll_pitch_yaw_from_matrix(
    matrix: np.ndarray, order: str = EULER_ORDER, degrees: bool = True
) -> np.ndarray:
    """Find the roll, pitch, and yaw angles from the given matrix.

    :param matrix: A 3x3 rotation matrix.
    :param order: The order of Euler angles (e.g., XYZ, ZYX).
    :param degrees: If True, return angles in degrees, otherwise in radians.
    :return: Roll, pitch, and yaw angles in the specified order and units.
    """
    return Rot.from_matrix(cls_1=np.ndarray, matrix=matrix).as_euler(order, degrees)


def align_to_gravity(
    g_vector: np.ndarray, order: str = EULER_ORDER, degrees: bool = True
) -> np.ndarray:
    """Find the best roll, pitch, and yaw angles that align with the gravity vector.

    :param g_vector: acceleration values in m/s^2
    :param order: The order of Euler angles (e.g., XYZ, ZYX).
    :param degrees: If True, return angles in degrees, otherwise in radians.
    :return: Rotation matrix that best aligns with gravity
    """
    x0 = np.zeros(3)
    residual = minimize(
        orientation_error,
        x0,
        method="nelder-mead",
        args=g_vector,
        options={"xatol": 1e-8, "disp": True},
    )
    return Rot.from_euler(seq=order, angles=residual.x, degrees=degrees).as_matrix()


def orientation_error(
    angles: np.ndarray,
    g_vector: np.ndarray,
    order: str = EULER_ORDER,
    degrees: bool = True,
) -> float:
    """Find the orientation that would best align with the gravity vector.

    :param angles: Roll, pitch, and yaw angles in degrees
    :param g_vector: Gravity vector
    :param order: The order of Euler angles (e.g., XYZ, ZYX).
    :param degrees: If True, angles are in degrees, otherwise in radians.
    :return: Error between the gravity vector and the projected vector in the m/s^2
    """
    gravity = np.linalg.norm(g_vector)
    rotation = Rot.from_euler(order, angles, degrees=degrees).as_matrix()
    error = np.linalg.norm(g_vector - gravity * rotation[2, :])
    return float(error)


def matrix_exponential(matrix: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Calculate the matrix exponential of a given matrix.

    :param matrix: A square matrix represented as a numpy array.
    :param t: The time parameter.
    :return: The matrix exponential of the given matrix.
    """
    if np.shape(matrix)[0] != np.shape(matrix)[1]:
        dim = matrix.shape
        msg = f"Input matrix must be square. Matrix has dimensions: {dim[0]}x{dim[1]}."
        logger.error(msg)
        raise ValueError(msg)

    mat = Matrix(matrix)
    if mat.is_diagonalizable():
        eig_val, eig_vec = np.linalg.eig(matrix)
        diagonal = np.diag(np.exp(eig_val * t))
        matrix_exp = eig_vec @ diagonal @ np.linalg.inv(eig_vec)
    else:
        P, J = mat.jordan_form()
        P, J = np.array(P).astype(np.float64), np.array(J).astype(np.float64)
        J = scipy.linalg.expm(t * J)
        matrix_exp = P @ J @ np.linalg.inv(P)
    return matrix_exp.real


def symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix.

    :param matrix: A square matrix represented as a numpy array.
    """
    if np.shape(matrix)[0] != np.shape(matrix)[1]:
        dim = matrix.shape
        msg = f"Input matrix must be square. Matrix has dimensions: {dim[0]}x{dim[1]}."
        logger.error(msg)
        raise ValueError(msg)

    return (matrix + matrix.T) / 2


def apply_angular_velocity(
    matrix: np.ndarray, omegas: np.ndarray, dt: float
) -> np.ndarray:
    """Apply angular velocity vector to a rotation matrix.

    :param matrix: A 3x3 rotation matrix.
    :param omegas: Angular velocity vector represented as a numpy array.
    :param dt: Time interval in seconds.
    :return: Updated rotation matrix and new angular velocity vector.
    """
    omega_exp = matrix_exponential(skew_matrix(omegas), t=dt)
    return matrix @ omega_exp


def apply_linear_acceleration(
    accel: np.ndarray,
    rotation_matrix: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply linear velocity vector to a rotation matrix, position, and velocity.

    :param accel: Linear acceleration vector represented as a numpy array.
    :param rotation_matrix: A 3x3 rotation matrix.
    :param position: Current position vector represented as a numpy array.
    :param velocity: Current velocity vector represented as a numpy array.
    :param dt: Time interval in seconds.
    :return: Updated position and velocity vectors.
    """
    residual = accel - grav * rotation_matrix @ np.array([[0], [0], [1]])
    velocity += residual * dt
    position += velocity * dt
    return position, velocity


if __name__ == "__main__":
    euler_order = "XYZ"
    grav = 9.81

    # define the full state
    rot = np.eye(3)
    rpy = Rot.from_matrix(matrix=rot).as_euler(euler_order, degrees=True)

    vel = np.zeros((3, 1))
    pos = np.zeros((3, 1))

    # record the measurements
    acc = np.zeros((3, 1))
    omega = np.zeros((3, 1))

    # process measurements
    rot = apply_angular_velocity(matrix=rot, omegas=omega, dt=0.01)
    pos, vel = apply_linear_acceleration(
        accel=acc, rotation_matrix=rot, position=pos, velocity=vel, dt=0.01
    )
