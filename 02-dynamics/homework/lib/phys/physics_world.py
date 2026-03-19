from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable
import numpy as np

from lib.basic_structs import Vec3
from lib.phys.phys_objects import RigidBody
from lib.phys.forces import ForceBase
from lib.phys.constraints.manager import ConstraintsManager
from lib.phys.ode_solvers import ODESolverBase
from lib.phys.collisions.collision_detector import CollisionDetectorBase
from lib.phys.contact_forces.handler import ContactHandlerBase

@dataclass
class PhysicsEngine:
    bodies: List[RigidBody]
    dynamics_solver: ODESolverBase
    forces: Optional[Dict[str, ForceBase]] = field(default_factory=lambda: {})
    constraints_manager: Optional[ConstraintsManager] = None
    collision_detector: Optional[CollisionDetectorBase] = None
    contact_forces_handler: Optional[ContactHandlerBase] = None
    enable_collisions: bool =  True
    callbacks: List[Callable] = field(default_factory=lambda: [])
    time: float = 0

    def gather_system_state(self):
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

    def zero_forces(self):
        for body in self.bodies:
            body.force_accumulator = Vec3(0, 0, 0)
            body.torque_accumulator = Vec3(0, 0, 0)

    def apply_contact_forces(self, t0, t1):
        collisions = []
        if not self.collision_detector is None:
            collisions = self.collision_detector.get_collisions()

            if not self.contact_forces_handler is None:
                self.contact_forces_handler.step(collisions, t0, t1)
        return collisions


    def apply_forces(self):
        for key, force in self.forces.items():
            force.apply_force(self.bodies)

    def apply_constraints(self, t0, t1):
        if not self.constraints_manager is None:
            self.constraints_manager.step(t0, t1)
    
    def step(self, t0, t1):
        self.time = t0

        self.zero_forces()
        
        if self.enable_collisions:
            collisions = self.apply_contact_forces(t0, t1)

        self.apply_forces()
        self.apply_constraints(t0, t1)

        x0 = self.gather_system_state()
        xt = self.dynamics_solver.ode(x0, t0, t1, dx_dt_func=self.system_dx_dt_func)
        self.set_system_state(xt)

        self.time = t1

        for callback in self.callbacks:
            callback(self)

    def system_dx_dt_func(self, x_system_state):
        ix_x = 0
        ix_p = 0
        system_dx_dt = {'x_dot': [], 'p_dot': []}
        for body in self.bodies:
            body_state = {
                'x': x_system_state['x'][ix_x: ix_x + body.state_x_size],
                'p': x_system_state['p'][ix_p: ix_p + body.state_p_size]
            }
            body_dx_dt = body.dx_dt_func(body_state)
            system_dx_dt['x_dot'] = np.concatenate((system_dx_dt['x_dot'], body_dx_dt['x_dot']))
            system_dx_dt['p_dot'] = np.concatenate((system_dx_dt['p_dot'], body_dx_dt['p_dot']))
            ix_x += body.state_x_size
            ix_p += body.state_p_size
        return system_dx_dt