from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from pyquaternion.quaternion import Quaternion

from scipy.integrate import solve_ivp

class Vec3(np.ndarray):
    '''
    a = Vec3([1, 2, 3])               # constructor from list
    b = Vec3(2, 3, 4)                 # constructor from args
    c = Vec3(np.array([3, 4.5, 2.2])) # ...

    d = Vec3([2, 2, 4, 5])            # ValueError
    print(c.x == c[0])                # True
    '''
    def __new__(cls, *args, info=None):
        if len(args) == 1:
            args = args[0]
        obj = np.asarray(args).astype(np.float32).view(cls)
        obj.info = info

        if obj.shape != (3,):
            raise ValueError(f"Vector must be shape (3,), given: {obj.shape}")
        
        return obj
    
    @property
    def x(self):
        return self.__getitem__(0)
    
    @x.setter
    def x(self, value):
        self.__setitem__(0, value)
    
    @property
    def y(self):
        return self.__getitem__(1)
    
    @y.setter
    def y(self, value):
        self.__setitem__(1, value)
    
    @property
    def z(self):
        return self.__getitem__(2)
    
    @z.setter
    def z(self, value):
        self.__setitem__(2, value)


@dataclass
class Pose:
    p: Vec3 = Vec3(0, 0, 0)                       # position of COM
    q: Quaternion = Quaternion(1, 0, 0, 0)        # quaternion

    @classmethod
    def from_pq(cls, p=[0, 0, 0], q=[1, 0, 0, 0]):
        return cls(p = Vec3(p), q = Quaternion(q))
        
    @classmethod
    def from_raw(cls, pose_raw = [0, 0, 0, 1, 0, 0, 0]):
        pose_raw = np.asarray(pose_raw)
        if pose_raw.shape != (7,):
            raise ValueError(f"Invalid raw shape: Input must be shape (7,), given: {pose_raw.shape}")
        p = pose_raw[:3]
        q = pose_raw[3:]
        return cls.from_pq(p=p, q=q)

    @classmethod
    def from_matrix(cls, matrix):
        try:
            shape = matrix.shape
        except AttributeError:
            raise TypeError("Invalid matrix type: Input must be a 4x4 numpy array or matrix")
        
        if shape != (4, 4):
            raise ValueError("Invalid matrix shape: Input must be a 4x4 numpy array or matrix")
        
        return cls(p = matrix[:3, 3], q = Quaternion._from_matrix(matrix))

    @property
    def matrix(self):
        m = self.q.transformation_matrix
        m[:3, 3] = self.p
        return m
    
    @property
    def raw(self):
        return np.concatenate((self.p, self.q.q))
    
    def inv(self):
        return Pose(
            p = - self.q.inverse.rotate(self.p),
            q = self.q.inverse
        )
    
    def __mul__(self, other):
        q = self.q * other.q
        p = self.q.rotate(other.p) + self.p
        return Pose.from_pq(p, q)
    
    def __rmul__(self, other):
        q = other.q * self.q
        p = other.q.rotate(self.p) + other.q
        return Pose.from_pq(p, q)
        

class ODESolverBase(ABC):
    @abstractmethod
    def ode(self, x_0, t_0, t_1, dx_dt_func: Callable):
        raise NotImplementedError


@dataclass
class PhysObject(ABC):
    name: str

    @property
    @abstractmethod
    def state_x_size(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def state_p_size(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def get_state(self):
        raise NotImplementedError


class ForceBase(ABC):
    @abstractmethod
    def apply_force(self, bodies: Dict[str, PhysObject]):
        pass

def vec_star(vec: Vec3):
    x, y, z = vec.tolist()
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])