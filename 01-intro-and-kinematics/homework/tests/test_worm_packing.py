import sys
from pathlib import Path

import numpy as np
import pytest

_hw = Path(__file__).resolve().parent.parent
if str(_hw) not in sys.path:
    sys.path.insert(0, str(_hw))

from solutions.worm_packing import JOINT_LIMIT_RADIUS, optimal_worm_config

LINK_LENGTHS_CASES = [
    np.array([2.0, 2.0, 2.0]),
    np.array([1.0, 2.0, 1.0, 2.0]),
    np.array([1.5, 1.5]),
]

TOL = 1e-5
COLLISION_MARGIN = 2.0 - 1e-4


def _segment_endpoints(link_lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    import torch
    import pytorch_kinematics as pk

    N = len(link_lengths)
    r_max = JOINT_LIMIT_RADIUS
    parts = ['<?xml version="1.0"?><robot name="worm">']
    parts.append('<link name="base_link"><inertial><mass value="0.001"/><inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial></link>')
    L0 = float(link_lengths[0])
    parts.append(f'<joint name="base_to_link0" type="fixed"><parent link="base_link"/><child link="link0"/>')
    parts.append('<origin xyz="0 0 0" rpy="0 0 0"/></joint>')
    parts.append(f'<link name="link0"><inertial><origin xyz="0 0 {L0/2:.6f}" rpy="0 0 0"/><mass value="0.1"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial></link>')
    for i in range(N - 1):
        L = float(link_lengths[i + 1])
        L_prev = float(link_lengths[i])
        parts.append(f'<joint name="j{i}a" type="revolute"><parent link="link{i}"/><child link="link{i}_mid"/>')
        parts.append(f'<origin xyz="0 0 {L_prev:.6f}" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-{r_max:.6f}" upper="{r_max:.6f}" effort="1" velocity="1"/></joint>')
        parts.append(f'<link name="link{i}_mid"><inertial><mass value="0.001"/><inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial></link>')
        parts.append(f'<joint name="j{i}b" type="revolute"><parent link="link{i}_mid"/><child link="link{i+1}"/>')
        parts.append(f'<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-{r_max:.6f}" upper="{r_max:.6f}" effort="1" velocity="1"/></joint>')
        parts.append(f'<link name="link{i+1}"><inertial><origin xyz="0 0 {L/2:.6f}" rpy="0 0 0"/><mass value="0.1"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial></link>')
    parts.append('</robot>')
    urdf = "".join(parts)
    chain = pk.build_chain_from_urdf(urdf.encode("utf-8"))
    th = {f"j{i}a": torch.tensor(float(angles[2 * i]), dtype=torch.float32) for i in range(N - 1)}
    th.update({f"j{i}b": torch.tensor(float(angles[2 * i + 1]), dtype=torch.float32) for i in range(N - 1)})
    ret = chain.forward_kinematics(th)
    endpoints = np.zeros((N + 1, 3))
    for i in range(N):
        T = ret[f"link{i}"].get_matrix()[0].numpy()
        endpoints[i + 1] = T[:3, 3] + link_lengths[i] * T[:3, 2]
    return endpoints


def _aabb_diagonal(endpoints: np.ndarray) -> float:
    lo, hi = endpoints.min(axis=0), endpoints.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def _segment_pair_distance(a0, a1, b0, b1) -> float:
    da, db = a1 - a0, b1 - b0
    dd = a0 - b0
    ada = np.dot(da, da) + 1e-10
    bdb = np.dot(db, db) + 1e-10
    adb = np.dot(da, db)
    add_ = np.dot(da, dd)
    bdd = np.dot(db, dd)
    den = ada * bdb - adb * adb
    t = np.clip((add_ * bdb - bdd * adb) / den, 0, 1)
    s = np.clip((bdd * ada - add_ * adb) / den, 0, 1)
    return float(np.linalg.norm(a0 + t * da - (b0 + s * db)))


@pytest.mark.parametrize("link_lengths", LINK_LENGTHS_CASES)
def test_compactness_improves(link_lengths: np.ndarray) -> None:
    angles = optimal_worm_config(link_lengths)
    N = len(link_lengths)
    n_joints = N - 1
    assert angles.shape == (2 * n_joints,), f"Expected shape {(2*n_joints,)}, got {angles.shape}"

    # joint limits
    for i in range(n_joints):
        t1, t2 = angles[2 * i], angles[2 * i + 1]
        assert t1**2 + t2**2 <= JOINT_LIMIT_RADIUS**2 + TOL, f"Joint {i} violates circular limit"

    endpoints = _segment_endpoints(link_lengths, angles)

    # no self-collision between non-consecutive links
    for i in range(N):
        for j in range(i + 2, N):
            d = _segment_pair_distance(endpoints[i], endpoints[i + 1], endpoints[j], endpoints[j + 1])
            assert d >= COLLISION_MARGIN, f"Pair ({i},{j}) distance {d:.4f} < {COLLISION_MARGIN}"

    # AABB must be smaller than the straight-line (zero-angle) configuration
    diagonal = _aabb_diagonal(endpoints)
    straight_diagonal = float(np.sum(link_lengths))
    print(f"  link_lengths={link_lengths.tolist()}  AABB diagonal={diagonal:.4f}  straight={straight_diagonal:.4f}")
    assert diagonal < straight_diagonal - TOL, (
        f"AABB diagonal {diagonal:.4f} not smaller than straight-line {straight_diagonal:.4f}"
    )
