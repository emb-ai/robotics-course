import numpy as np

JOINT_LIMIT_RADIUS = np.pi / 3


def optimal_bead_config(
    link_lengths: np.ndarray
    ) -> np.ndarray:
    """
    Compute joint angles (in radians) that minimize the bounding sphere radius of the articulated chain.

    Parameters
    ----------
    link_lengths : array-like of shape (N,)
        Lengths of each link in the chain.

    Returns
    -------
    ndarray of shape (N-1, 2)
        Joint angles [theta1, theta2] at each joint, satisfying:
        - Spherical cap: great-circle angle from identity ≤ π/3, i.e. cos(θ₁)cos(θ₂) ≥ cos(π/3) (joint limits)
        - No overlap: Spheres of radius 1 at each joint do not intersect.

    Notes
    -----
    The returned configuration achieves maximal compactness (minimal enclosing sphere), respects joint limits,
    and maintains collision-free geometry between beads.
    """
    raise NotImplementedError
