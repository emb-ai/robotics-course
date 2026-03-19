from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import cg

from lib.phys.phys_objects import RigidBody
from lib.phys.constraints.constraints import Constraint

@dataclass
class MatrixCell:
    i_start: int = None
    j_start: int = None
    i_end: int = None
    j_end: int = None

    def is_valid(self):
        if self.i_start is None or self.i_end is None or self.j_start is None or self.j_end is None:
            return False
        if self.i_start >= self.i_end:
            return False
        if self.j_start >= self.j_end:
            return False
        return True
    
    @property
    def slice_shape(self):
        i_size = 0
        if not (self.i_start is None or self.i_end is None):
            i_size = self.i_end - self.i_start
        j_size = 0
        if not (self.j_start is None or self.j_end is None):
            j_size = self.j_end - self.j_start
        return (i_size, j_size)
    

def _concatenate(values):
    if len(np.asarray(values).shape) == 1:
        return np.asarray(values)
    return np.concatenate(values)

@dataclass
class ConstraintsManager:
    bodies: List[RigidBody]
    constraints: Dict[str, Constraint]
    k_s: float = 10000
    k_d: float = 100
    
    def __post_init__(self):
        self.J, self.J_idxs = self._construct_empty_jacobian()
        self.J_num_rows, self.J_num_cols = self.J.shape

        self.W, self.W_idxs = self._construct_system_inv_inertia()


    def add_new_constraint(self, constraint_name: str, constraint: Constraint):
        self.constraints[constraint_name] = constraint

        self.J_idxs[constraint_name] = {}

        row = self.J_num_rows
        col = 0
        for body in self.bodies:
            self.J_idxs[constraint_name][body.name] = MatrixCell(
                    i_start = row, j_start = col,
                    i_end = row + constraint.dim_size,
                    j_end = col + body.state_p_size
                )
            col += body.state_p_size

        self.J_num_rows += constraint.dim_size

        new_J = csc_array((self.J_num_rows, self.J_num_cols), dtype=np.float32)
        new_J[:-constraint.dim_size, :] = self.J
        self.J = new_J


    def _construct_empty_jacobian(self):
        row = 0
        col = 0
        J_idxs = {} # dicts of dicts; J_idxs[constraint_name][body] -> idxs in J

        for body in self.bodies:
            row = 0

            for constraint_name, constraint in self.constraints.items():
                if not constraint_name in J_idxs:
                    J_idxs[constraint_name] = {}    

                J_idxs[constraint_name][body.name] = MatrixCell(
                    i_start = row, j_start = col,
                    i_end = row + constraint.dim_size,
                    j_end = col + body.state_p_size
                )
                row += constraint.dim_size

            col += body.state_p_size

        return csc_array((row, col), dtype=np.float32), J_idxs


    def _construct_system_inv_inertia(self):
        W = csc_array((self.J_num_cols, self.J_num_cols), dtype=np.float32)
        W_idxs = {}
        i, j = 0, 0
        for body in self.bodies:
            idx = MatrixCell(i, j, i + body.state_p_size, j + body.state_p_size)
            W_idxs[body.name] = idx

            # fill inv masses since they are constant
            W[idx.i_start : idx.i_start + 3, idx.j_start : idx.j_start + 3] = body.inv_mass * np.eye(3)
            i += body.state_p_size
            j += body.state_p_size
        return W, W_idxs
    
    def update_W(self):
        for body in self.bodies:
            idx = self.W_idxs[body.name]
            assert idx.is_valid()
            if idx.slice_shape[1] == 3: # skip particle update
                continue
            self.W[idx.i_start + 3 : idx.i_end,
                   idx.j_start + 3 : idx.j_end] = body.inertia_tensor_inv
            
    def update_J(self):
        for constraint_name, constraint in self.constraints.items():
            updates = constraint.get_J_updates()
            for body_name, J_update in updates.items():
                idx = self.J_idxs[constraint_name][body_name]
                assert idx.is_valid()
                assert idx.slice_shape == J_update.shape
                self.J[idx.i_start : idx.i_end,
                   idx.j_start : idx.j_end] = J_update
                
    def get_C(self):
        C = []
        for _, constraint in self.constraints.items():
            C.append(constraint.get_C())
        C = _concatenate(C)
        assert C.shape[0] == self.J_num_rows
        return C
    
    def get_J_dot_V(self):
        J_dot_V = []
        for _, constraint in self.constraints.items():
            J_dot_V.append(constraint.get_J_dot_V())
        J_dot_V = _concatenate(J_dot_V)
        assert J_dot_V.shape[0] == self.J_num_rows
        return J_dot_V

    def get_V(self):
        V = []
        for body in self.bodies:
            V.append(body.body_velocity)
            if body.state_p_size == 6:
                V.append(body.angular_velocity)
        V = _concatenate(V)
        assert V.shape[0] == self.J_num_cols
        return V
    
    def get_F(self):
        F = []
        for body in self.bodies:
            F.append(body.force_accumulator)
            if body.state_p_size == 6:
                F.append(body.torque_accumulator)
        F = _concatenate(F)
        assert F.shape[0] == self.J_num_cols
        return F

    def update_constraints(self, t_0, t_1):
        for _, constraint in self.constraints.items():
            constraint.update(t_0, t_1)

    def calc_forces(self, t_0, t_1):
        if len(self.constraints) == 0:
            return None
        self.update_constraints(t_0, t_1)
        self.update_W()
        W = self.W
        self.update_J()
        J = self.J
        J_dot_V = self.get_J_dot_V()
        C = self.get_C()
        V = self.get_V()
        F = self.get_F()
        K = J @ W @ J.T
        rhs = -J_dot_V - J @ W @ F - self.k_d * J @ V - self.k_s * C
        l, exit_code = cg(K, rhs, atol=1e-5)
        assert exit_code == 0
        F_C = J.T @ l
        return F_C

    def step(self, t_0, t_1):
        F_C = self.calc_forces(t_0, t_1)
        if not F_C is None:
            self.update_forces(F_C)

    def update_forces(self, F_C):
        i = 0
        for body in self.bodies:
            f_c = F_C[i : i + 3]
            body.force_accumulator += f_c
            i += 3
            if body.state_p_size == 6:
                tau_c = F_C[i : i + 3]
                body.torque_accumulator += tau_c
                i += 3
