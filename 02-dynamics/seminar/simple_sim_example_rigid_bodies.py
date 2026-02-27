from typing import List, Optional, Union
from dataclasses import dataclass, field
from argparse import ArgumentParser

import numpy as np
from pyquaternion.quaternion import Quaternion

import vpython as vp

from simple_sim_tutorial import *
from simple_sim_example_particles import Particle, PhysicsEngine


class EulerMethod(ODESolverBase):
    '''
    s_0 = (x_0, v_0)
    dx_dt_func = (f(x, v), g(x, v))

    Step:
    x_1 = x_0 + dt * f(x_0, v_0)
    v_1 = v_0 + dt * g(x_0, v_0)
    '''
    def ode(self, x_0, t_0, t_1, dx_dt_func):
        dt = t_1 - t_0 # stepsize
        f = dx_dt_func(x_0)
        x_1 = x_0['x'] + dt * f['x_dot']
        p_1 = x_0['p'] + dt * f['p_dot']
        return {
            'x': x_1,
            'p': p_1
        }

@dataclass
class RigidBody(PhysObject):
    # constants
    inv_mass: float                            # 1 / m
    body_inertia_tensor_inv: np.array          # I_body ^ -1

    # state
    pose: Pose = Pose()                        # x(t), q(t)
    linear_momentum: Vec3 = Vec3(0, 0, 0)      # P(t)
    angular_momentum: Vec3 = Vec3(0, 0, 0)     # L(t)

    # computed quantites
    force_accumulator: Vec3 = Vec3(0, 0, 0)    # F(t) 
    torque_accumulator: Vec3 = Vec3(0, 0, 0)   # tau(t) 

    def zero_forces(self):
        self.force_accumulator = Vec3(0, 0, 0)
        self.torque_accumulator = Vec3(0, 0, 0)
    
    # derived auxiliary variables
    # Note: we make separate functions for each property, which use only 
    # body constants (but no state variables) for calculation 
    # (i.e. `self.body_velocity` and `self.calc_velocity(...)`, 
    # `self.inertia_tensor_inv` and `self.calc_inertia_tensor_inv(...)`, etc.)
    # These calc_* functions are need for `self.dx_dt_func`.
    @property                                  # P(t) / m
    def body_velocity(self):
        return self.calc_velocity(self.linear_momentum)
    
    def calc_velocity(self, linear_momentum):
        return self.inv_mass * linear_momentum
    
    
    @property                                  # R(t)
    def rotation_matrix(self):
        return self.pose.q.rotation_matrix
    
    
    @property
    def inertia_tensor_inv(self):              # I(t) ^ -1
        return self.calc_inertia_tensor_inv(self.pose.q)

    def calc_inertia_tensor_inv(self, q: Quaternion):
        rotation_matrix = q.rotation_matrix
        return rotation_matrix @ self.body_inertia_tensor_inv @ rotation_matrix.T
    

    @property
    def angular_velocity(self):                # omega(t)
        return self.calc_angular_velocity(self.pose.q, self.angular_momentum)
    
    def calc_angular_velocity(self, q: Quaternion, angular_momentum: Vec3):
        inertia_tensor_inv = self.calc_inertia_tensor_inv(self.pose.q)
        return inertia_tensor_inv @ angular_momentum
    
    # return state
    def get_state(self):
        state = dict(
            x = self.pose.raw,
            p = np.concatenate((self.linear_momentum, self.angular_momentum))
        )
        return state

    # set state
    def set_state(self, new_state):
        assert len(new_state['x']) == self.state_x_size
        assert len(new_state['p']) == self.state_p_size
        self.pose = Pose.from_raw(new_state['x'])
        self.linear_momentum = Vec3(new_state['p'][:3])
        self.angular_momentum = Vec3(new_state['p'][3:])

    # 3 for particles, 7 for rigid bodies
    @property
    def state_x_size(self):
        return len(self.get_state()['x'])

    # 3 for particles, 6 for rigid bodies
    @property
    def state_p_size(self):
        return len(self.get_state()['p'])

    @property
    def state_size(self):
        return self.state_x_size + self.state_p_size

    def dx_dt_func(self, x_state):
        x = x_state['x'][:3]
        q = x_state['x'][3:]
        linear_momentum = x_state['p'][:3]
        angular_momentum = x_state['p'][3:]

        body_velocity = self.calc_velocity(linear_momentum)
        angular_velocity = self.calc_angular_velocity(q, angular_momentum)

        q_dot = 0.5 * Quaternion(vector=angular_velocity) * q
        P_dot = self.force_accumulator
        L_dot = self.torque_accumulator

        dx_dt = {}
        dx_dt['x_dot'] = np.concatenate((body_velocity, q_dot.q))
        dx_dt['p_dot'] = np.concatenate((P_dot, L_dot))
        
        return dx_dt
        

@dataclass
class HookeForce(ForceBase):
    body_a: Union[Particle, RigidBody]
    body_b: Union[Particle, RigidBody]
    ks: float
    kd: float
    application_point_a: Vec3 = Vec3(0, 0, 0)
    application_point_b: Vec3 = Vec3(0, 0, 0)
    rest_length: float = 1.0

    def calc_force(self):
        com_a_pose = Pose(self.body_a.pos) if isinstance(self.body_a, Particle) else self.body_a.pose
        com_b_pose = Pose(self.body_b.pos) if isinstance(self.body_b, Particle) else self.body_b.pose
    
        a = (com_a_pose * Pose(self.application_point_a)).p
        b = (com_b_pose * Pose(self.application_point_b)).p
        l = a - b

        r_a = a - com_a_pose.p
        r_b = b - com_b_pose.p

        a_vel = self.body_a.body_velocity
        if isinstance(self.body_a, RigidBody):
            a_vel += np.cross(self.body_a.angular_velocity, r_a)
            
        b_vel = self.body_b.body_velocity 
        if isinstance(self.body_b, RigidBody):
            b_vel += np.cross(self.body_b.angular_velocity, r_b)
            
        l_dot = a_vel - b_vel

        delta_l = np.linalg.norm(l) - self.rest_length
        delta_l_dot = np.dot(l_dot, l) / (np.linalg.norm(l) + 1e-3)

        module = self.ks * delta_l + self.kd * delta_l_dot
        return -module * l / (np.linalg.norm(l) + 1e-3)

    def apply_force(self, bodies):
        hooke_force = self.calc_force()

        self.body_a.force_accumulator += hooke_force
        self.body_b.force_accumulator += -hooke_force

        if isinstance(self.body_a, RigidBody):
            r_a = (self.body_a.pose * Pose(self.application_point_a)).p - self.body_a.pose.p
            torque_a = np.cross(r_a, hooke_force)
            self.body_a.torque_accumulator += torque_a

        if isinstance(self.body_b, RigidBody):
            r_b = (self.body_b.pose * Pose(self.application_point_b)).p - self.body_b.pose.p
            torque_b = np.cross(r_b, -hooke_force)
            self.body_b.torque_accumulator += torque_b

class GravityForce(ForceBase):
    def __init__(self, g_vector:Vec3 = Vec3(0, -9.81, 0)):
        self.g_vector = g_vector

    def apply_force(self, bodies: List[Particle]):
        for body in bodies:
            if body.inv_mass > 0:
                body.force_accumulator += self.g_vector / body.inv_mass

@dataclass
class Actor:
    name: str 
    phys_obj: RigidBody
    visual: vp.baseObj

    def __post_init__(self):
        self.update_visuals()
    
    def update_visuals(self):
        if isinstance(self.phys_obj, Particle):
            # particle
            self.visual.pos = vp.vector(*self.phys_obj.pos.tolist())
        else:
            # rigid body
            self.visual.pos = vp.vector(*self.phys_obj.pose.p.tolist())
    
            ox = self.phys_obj.pose.matrix[:3, 0]
            oy = self.phys_obj.pose.matrix[:3, 1]
    
            self.visual.axis = vp.vector(*ox.tolist()) * self.visual.length
            self.visual.up = vp.vector(*oy.tolist()) * self.visual.height


@dataclass
class VisualLinkageBodies:
    name: str
    actor_a: Actor
    application_point_a: Vec3
    actor_b: Actor
    application_point_b: Vec3
    visual: vp.baseObj
    
    def __post_init__(self):
        self.update_visuals()

    def update_visuals(self):
        start = vp.vector(*(self.actor_a.phys_obj.pose * Pose(self.application_point_a)).p.tolist())
        end = vp.vector(*(self.actor_b.phys_obj.pose * Pose(self.application_point_b)).p.tolist())
        axis = end - start

        self.visual.pos = start
        self.visual.axis = axis

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


# calculate inertia tensor I_body for a box
def get_box_inertia(mass: int, extents: tuple):
    x, y, z = extents
    return mass / 12 * np.array(
        [
            [y ** 2 + z ** 2, 0, 0],
            [0, x ** 2 + z ** 2, 0],
            [0, 0, x ** 2 + y ** 2]
        ]
    )

def make_box_actor(name,
        extents=(1, 1, 1),
        color = vp.color.red,
        pos = [0, 0, 0],
        lin_momentum = [0, 0, 0],
        ang_momentum = [0, 0, 0],
        mass = 1.0
):
    body_inertia_tensor = get_box_inertia(mass, extents)
    return Actor(
        name = name,
        phys_obj=RigidBody(
            name = name,
            inv_mass = 1 / mass,
            body_inertia_tensor_inv = np.linalg.inv(body_inertia_tensor),
            pose = Pose.from_pq(p = pos),
            linear_momentum = Vec3(lin_momentum),
            angular_momentum = Vec3(ang_momentum)
        ), 
        visual=vp.box(color = color, size=vp.vector(*extents), make_trail=False)
    )

def example_bodies_spring():
    # rigid bodies on a spring
    scene = vp.canvas()

    extents1 = np.array([1, 1, 1])
    box1 = make_box_actor('box1', extents1,
                            pos=[3, 0, 0], mass=1.0)
    extents2 = np.array([2, 3, 5])
    box2 = make_box_actor('box2', extents2,
                            pos=[-3, 0, 0])

    spring = VisualLinkageBodies('spring',  box1, extents1 / 2, box2, extents2 / 2,
                        visual=vp.helix(color = vp.color.green, radius=0.4, thickness=0.1),)

    forces = [
        HookeForce(box1.phys_obj, box2.phys_obj, application_point_a=extents1 / 2, 
                application_point_b=extents2 / 2,
                ks=0.8, kd=0.0, rest_length=5.0)
    ]

    dt = 0.01
    sim_world = SimWorld(dt, 200, entities = [box1, box2, spring],
                        forces = forces,
                        dynamics_solver=EulerMethod())
    sim_world.run(n_steps=2000)

    print("FINISHED")
    print("Close tab in browser to exit")

def example_hanging_particle():
    print("rigid body hanging on a spring")
    scene = vp.canvas(title='rigid body hanging on a spring')

    extents1 = np.array([1, 1, 1])
    box1 = make_box_actor('box1', extents1,
                            pos=[3, 0, 0], mass=1.0)
    box1.phys_obj.inv_mass = 0
    box1.phys_obj.body_inertia_tensor_inv = np.zeros((3, 3))

    extents2 = np.array([2, 3, 5])
    box2 = make_box_actor('box2', extents2,
                            pos=[-3, 0, 0])

    spring = VisualLinkageBodies('spring',  box1, extents1 / 2, box2, extents2 / 2,
                        visual=vp.helix(color = vp.color.green, radius=0.4, thickness=0.1),)

    forces = [
        HookeForce(box1.phys_obj, box2.phys_obj, application_point_a=extents1 / 2, 
                application_point_b=extents2 / 2,
                ks=10.8, kd=2.0, rest_length=5.0),
        GravityForce()
    ]

    dt = 0.01
    sim_world = SimWorld(dt, 200, entities = [box1, box2, spring],
                        forces = forces,
                        dynamics_solver=EulerMethod())
    sim_world.run(n_steps=2000)

    print("FINISHED")
    print("Close tab in browser to exit")

EXAMPLES = {
    'bodies_spring': example_bodies_spring,
    'hanging_body': example_hanging_particle,
}

def parse_args():
    parser = ArgumentParser(description='Particle simulation examples')
    parser.add_argument('example', choices=EXAMPLES.keys(),
                        help='Which example to run')
    return parser.parse_args()

def main(args):
    EXAMPLES[args.example]()


if __name__ == '__main__':
    main(parse_args())