from typing import List, Optional, Union
from dataclasses import dataclass, field
from argparse import ArgumentParser

import numpy as np
from pyquaternion.quaternion import Quaternion
from scipy.sparse import csc_array
from scipy.sparse.linalg import cg

import vpython as vp

from simple_sim_tutorial import *
from simple_sim_example_rigid_bodies import RigidBody, Actor, GravityForce, EulerMethod, make_box_actor

class Constraint(ABC):
    @abstractmethod
    def update(self, t_0, t_1):    # update internal values
        pass

    @property
    @abstractmethod
    def constrained_bodies(self):  # return names of the bodies under constraint
        pass

    @property
    @abstractmethod                # number of rows in C
    def dim_size(self):
        pass

    @abstractmethod                # calculate C(r)
    def get_C(self):
        pass

    @abstractmethod
    def get_J_updates(self):       # calculate dJ/dq_i for each body under constraint
        pass
    

    @abstractmethod                # calculate  J' V
    def get_J_dot_V(self):
        pass


@dataclass
class MatrixCell:
    i_start: int = None
    j_start: int = None
    i_end: int = None
    j_end: int = None

    @property
    def slice_shape(self):
        i_size = 0
        if not (self.i_start is None or self.i_end is None):
            i_size = self.i_end - self.i_start
        j_size = 0
        if not (self.j_start is None or self.j_end is None):
            j_size = self.j_end - self.j_start
        return (i_size, j_size)

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


    def _construct_empty_jacobian(self):
        row = 0
        col = 0
        J_idxs = {} # dict of dicts; J_idxs[constraint_name][body] -> idxs in J

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

    def step(self, t_0, t_1):
        if len(self.constraints) == 0:
            return
        # calculate internal values in constraints
        self.update_constraints(t_0, t_1)

        # get sparse matrices values

        self.update_W()
        W = self.W # system invese inertia matrix
        
        self.update_J()
        J = self.J # constrains jacobian

        # get vectors
        J_dot_V = self.get_J_dot_V() # J'V
        C = self.get_C() # constrains values
        V = self.get_V() # system velocity 6N vector (for particles 3N)
        F = self.get_F() # system force 6N vector (for particles 3N)

        # calculate K=JWJ^T
        K = J @ W @ J.T

        rhs = -J_dot_V - J @ W @ F - self.k_d * J @ V - self.k_s * C
        # solve using conjugade gradients
        l, exit_code = cg(K, rhs, atol=1e-5)
        assert exit_code == 0

        F_C = J.T @ l

        # update forces (and torques)
        self.update_forces(F_C)
    
    def update_W(self):
        for body in self.bodies:
            idx = self.W_idxs[body.name]
            if idx.slice_shape[1] == 3: # skip particle update
                continue
            self.W[idx.i_start + 3 : idx.i_end,
                   idx.j_start + 3 : idx.j_end] = body.inertia_tensor_inv
            
    def update_J(self):
        for constraint_name, constraint in self.constraints.items():
            updates = constraint.get_J_updates()
            for body_name, J_update in updates.items():
                idx = self.J_idxs[constraint_name][body_name]
                assert idx.slice_shape == J_update.shape
                self.J[idx.i_start : idx.i_end,
                   idx.j_start : idx.j_end] = J_update
                
    def get_C(self):
        C = []
        for _, constraint in self.constraints.items():
            C.append(constraint.get_C())
        C = np.concatenate(C)
        assert C.shape[0] == self.J_num_rows
        return C
    
    def get_J_dot_V(self):
        J_dot_V = []
        for _, constraint in self.constraints.items():
            J_dot_V.append(constraint.get_J_dot_V())
        J_dot_V = np.concatenate(J_dot_V)
        assert J_dot_V.shape[0] == self.J_num_rows
        return J_dot_V

    def get_V(self):
        V = []
        for body in self.bodies:
            V.append(body.body_velocity)
            if body.state_p_size == 6:
                V.append(body.angular_velocity)
        V = np.concatenate(V)
        assert V.shape[0] == self.J_num_cols
        return V
    
    def get_F(self):
        F = []
        for body in self.bodies:
            F.append(body.force_accumulator)
            if body.state_p_size == 6:
                F.append(body.torque_accumulator)
        F = np.concatenate(F)
        assert F.shape[0] == self.J_num_cols
        return F

    def update_constraints(self, t_0, t_1):
        for _, constraint in self.constraints.items():
            constraint.update(t_0, t_1)


    def update_forces(self, F_C):
        i = 0
        for body in self.bodies:
            f_c = F_C[i : i + 3]
            body.force_accumulator += f_c

            i += 3

            if body.state_p_size == 6: # rigid body
                tau_c = F_C[i : i + 3]
                body.torque_accumulator += tau_c

                i += 3

@dataclass
class BallAndSocketPoint(Constraint):
    '''
    Body point with coordinates r_f_local in the body frame
    is fixed in space. The body is free to rotate around this fixed point.

    r is the COM position of the body. 
    At each timestep we transform r_f_local to r_f in the world frame.
    Initially, R := r_f at time t = 0 and is constant during the whole simulation.

    So the constraint uquation looks as follows:

    C(r) = r + r_f - R

    Args
    ----
    body: RigidBody
        Body under constraint.

    body_fixed_point_local: Vec3
        Body anchor point in body frame (r_f_local).
    '''
    body: RigidBody
    body_fixed_point_local: Vec3 # r_f_local

    def __post_init__(self,): # Calc R
        self.fixed_point_world = (self.body.pose * Pose(p=self.body_fixed_point_local)).p

    def update(self, t_0, t_1):
        '''
        find r_f from r_f_local
        '''
        self.body_fixed_point_world = self.body.pose.q.rotation_matrix @ self.body_fixed_point_local

    @property
    def constrained_bodies(self):
        return [self.body]

    @property
    def dim_size(self):
        return 3

    def get_C(self):
        '''
        C = r + r_f - R
        '''
        return self.body.pose.p + self.body_fixed_point_world - self.fixed_point_world 

    def get_J_updates(self):
        '''
        J = [ E | - [r_f]x ]
        '''
        r_f_star = vec_star(self.body_fixed_point_world)
        return {
            self.body.name: np.hstack((np.eye(3), -r_f_star))
        }
    
    def get_J_dot_V(self):
        '''
        J'V = omega x (omega x r_f)
        '''
        return np.cross(self.body.angular_velocity, np.cross(self.body.angular_velocity, self.body_fixed_point_world))

@dataclass
class PhysicsEngine:
    bodies: List[RigidBody]
    dynamics_solver: ODESolverBase
    forces: Optional[List[ForceBase]] = field(default_factory=lambda: [])
    constraints_manager: Optional[ConstraintsManager] = None

    def apply_constraints(self, t0, t1):
        self.constraints_manager.step(t0, t1)

    def step(self, t0, t1):
        self.zero_forces()
        self.apply_forces()
        self.apply_constraints(t0, t1)

        s0 = self.get_system_state()
        s1 = self.dynamics_solver.ode(s0, t0, t1, dx_dt_func=self.system_dx_dt_func)
        self.set_system_state(s1)

    def zero_forces(self):
        for body in self.bodies:
            body.zero_forces()

    def apply_forces(self):
        for force in self.forces:
            force.apply_force(self.bodies)

    def get_system_state(self):
        system_state = {}
        for body in self.bodies:
            if len(system_state) == 0:
                system_state = body.get_state()
            else:
                body_state = body.get_state()
                system_state['x'] = np.concatenate((system_state['x'], body_state['x']))
                system_state['p'] = np.concatenate((system_state['p'], body_state['p']))
        return system_state

    def set_system_state(self, new_state: Dict):
        ix_x = 0
        ix_p = 0
        for body in self.bodies:
            body_state = {
                'x': new_state['x'][ix_x: ix_x + body.state_x_size],
                'p': new_state['p'][ix_p: ix_p + body.state_p_size]
            }
            body.set_state(body_state)
            ix_x += body.state_x_size
            ix_p += body.state_p_size

    def system_dx_dt_func(self, system_state):
        ix_x = 0
        ix_p = 0
        system_dx_dt = {'x_dot': [], 'p_dot': []}
        for body in self.bodies:
            body_state = {
                'x': system_state['x'][ix_x: ix_x + body.state_x_size],
                'p': system_state['p'][ix_p: ix_p + body.state_p_size]
            }
            body_dx_dt = body.dx_dt_func(body_state)
            system_dx_dt['x_dot'] = np.concatenate((system_dx_dt['x_dot'], body_dx_dt['x_dot']))
            system_dx_dt['p_dot'] = np.concatenate((system_dx_dt['p_dot'], body_dx_dt['p_dot']))
            ix_x += body.state_x_size
            ix_p += body.state_p_size
        return system_dx_dt

class SimWorld:
    def __init__(self, dt, rate, entities = [], **kwargs):
        self.dt = dt
        self.rate = rate
        self.entities = entities
        
        phys_bodies = []
        self.entity_names = []
        self.phys_obj_names = []
        for entity in self.entities:
            if isinstance(entity, Actor):
                phys_bodies.append(entity.phys_obj)
                if entity.phys_obj.name in self.phys_obj_names:
                    raise ValueError(f"Duplicate entity name: {entity.phys_obj.name}")
                self.phys_obj_names.append(entity.phys_obj.name)
            if entity.name in self.entity_names:
                raise ValueError(f"Duplicate entity name: {entity.name}")
            self.entity_names.append(entity.name)

        self.phys_world = PhysicsEngine(
                    bodies = phys_bodies,
                    **kwargs
                )
        
    def update_visuals(self):
        for enity in self.entities:
            enity.update_visuals()

    def step(self, t_0, t_1):
        if len(self.entities) > 0:
            self.phys_world.step(t_0, t_1)
        self.update_visuals()

    def run(self, n_steps = None):
        elapsed_steps = 0
        t = 0
        while True:
            vp.rate(self.rate)
            self.step(t, t + self.dt)
            t += self.dt
            elapsed_steps += 1
            if n_steps is not None and elapsed_steps >= n_steps:
                break


def example():
    scene = vp.canvas(title='Fixed hinge')

    extents1 = np.array([2, 3, 15])
    box1 = make_box_actor('box1', extents1, color=vp.color.green,
                            pos=[0, 0, 0], mass=1.0)

    constraints = {
        "fixed_point1": BallAndSocketPoint(
            body=box1.phys_obj,
            body_fixed_point_local=Vec3(extents1 / 2), 
        ),
    }

    forces = [
        GravityForce()
    ]

    fixed_point = vp.sphere(pos=vp.vector(*(extents1 / 2).tolist()), radius=0.5, color = vp.color.red)

    dt = 0.01
    sim_world = SimWorld(dt, 200, entities = [box1],
                        forces = forces,
                        dynamics_solver=EulerMethod(),
                        constraints_manager=ConstraintsManager(bodies = [box1.phys_obj], constraints=constraints))
    sim_world.run(n_steps=2000)

    print("FINISHED")
    print("Close tab in browser to exit")


def main():
    example()


if __name__ == '__main__':
    main()