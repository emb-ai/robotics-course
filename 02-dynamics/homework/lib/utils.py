import numpy as np
from .basic_structs import Vec3
def skew_sym(vec: Vec3):
    x, y, z = vec.tolist()
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

