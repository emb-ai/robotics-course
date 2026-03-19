from dataclasses import dataclass
import numpy as np
from pyquaternion.quaternion import Quaternion


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
    p: Vec3 = Vec3(0, 0, 0)
    q: Quaternion = Quaternion(1, 0, 0, 0)

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



def test1():
    v = -Vec3(1, 2, 3)
    print(v)
    # print(v._x)

    s = Vec3([0, 1, 2]) + Vec3([4, 1, 2])
    s = s.astype(np.int32)
    print(s.dtype)
    print(s)

    print(np.dot(s, v))
    print(np.cross(s, v))

    v /= np.linalg.norm(v)
    print(v)
    print(v.x)
    
    v.z = 0.234
    print(v)

def test2():
    a = np.random.randn(7)
    # a /= np.linalg.norm(a)

    p1 = Pose.from_raw(a)
    print(p1)
    print(p1.inv())
    print(np.linalg.inv(p1.matrix))
    print(p1.inv().matrix)
    print("=====")
    print(p1.q.transformation_matrix)
    print(np.linalg.inv(p1.q.transformation_matrix.T))

if __name__ == '__main__':
    test2()