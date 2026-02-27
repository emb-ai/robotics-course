from typing import List, Optional, Union
from dataclasses import dataclass, field
from argparse import ArgumentParser

import numpy as np
from pyquaternion.quaternion import Quaternion

import vpython as vp

from simple_sim_tutorial import *


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
class Particle(PhysObject):
    # constants
    inv_mass: float                            # 1 / m

    # state
    pos: Vec3 = Vec3(0, 0, 0)                  # x(t)
    linear_momentum: Vec3 = Vec3(0, 0, 0)      # P(t)

    # derived auxiliary variables
    @property
    def body_velocity(self):                   # P(t) / m
        return self.calc_velocity(self.linear_momentum)

    # we make separate function which uses only body constants (but no state variables) 
    # for calculation each auxiliary variables
    def calc_velocity(self, linear_momentum):
        return linear_momentum * self.inv_mass
        
    # computed quantites
    force_accumulator: Vec3 = Vec3(0, 0, 0)    # F(t) 

    def zero_forces(self):
        self.force_accumulator = Vec3(0, 0, 0)

    # return state
    def get_state(self):
        state = dict(
            x = self.pos,
            p = self.linear_momentum
        )
        return state

    # set state
    def set_state(self, new_state):
        assert len(new_state['x']) == self.state_x_size
        assert len(new_state['p']) == self.state_p_size
        self.pos = Vec3(new_state['x'])
        self.linear_momentum = Vec3(new_state['p'])

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
        x = x_state['x']
        linear_momentum = x_state['p']

        body_velocity = self.calc_velocity(linear_momentum)
        return {
            'x_dot': body_velocity,
            'p_dot': self.force_accumulator
        }


class GravityForce(ForceBase):
    def __init__(self, g_vector:Vec3 = Vec3(0, -9.81, 0)):
        self.g_vector = g_vector

    def apply_force(self, bodies: List[Particle]):
        for body in bodies:
            if body.inv_mass > 0:
                body.force_accumulator += self.g_vector / body.inv_mass 

class HookeForce(ForceBase):
    def __init__(self, body_a: Particle, body_b: Particle,
                 ks: float, kd: float = 0., rest_length = 0):
        self.body_a = body_a
        self.body_b = body_b
        self.ks = ks
        self.kd = kd
        self.r = rest_length

    def calc_force(self):
        a = self.body_a.pos
        b = self.body_b.pos
        l = a - b

        a_vel = self.body_a.body_velocity
        b_vel = self.body_b.body_velocity
        l_dot = a_vel - b_vel

        delta_l = np.linalg.norm(l) - self.r
        delta_l_dot = np.dot(l_dot, l) / (np.linalg.norm(l) + 1e-3)

        module = self.ks * delta_l + self.kd * delta_l_dot
        return -module * l / (np.linalg.norm(l) + 1e-3)

    def apply_force(self, bodies):
        hooke_force = self.calc_force()

        self.body_a.force_accumulator += hooke_force
        self.body_b.force_accumulator += -hooke_force

@dataclass
class PhysicsEngine:
    bodies: List[Particle]
    dynamics_solver: ODESolverBase
    forces: Optional[List[ForceBase]] = field(default_factory=lambda: [])

    def step(self, t0, t1):
        self.zero_forces()
        self.apply_forces()

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
    
@dataclass
class Actor:
    name: str 
    phys_obj: Particle
    visual: vp.baseObj

    def __post_init__(self):
        self.update_visuals()
    
    def update_visuals(self):
        self.visual.pos = vp.vector(*self.phys_obj.pos.tolist())

def make_particle_actor(name, color=vp.color.red, radius = 0.2, pos=[0, 0, 0], lin_momentum=[0, 0, 0], mass=1.0):
    return Actor(
        name = name,
        phys_obj = Particle(
            name = name,
            inv_mass = 1 / mass,
            pos = Vec3(pos),
            linear_momentum = Vec3(lin_momentum)
        ),
        visual = vp.sphere(radius=radius, color = color)
    )

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

@dataclass
class VisualLinkageParticles:
    name: str
    actor_1: Actor
    actor_2: Actor
    visual: vp.baseObj
    
    def __post_init__(self):
        self.update_visuals()

    def update_visuals(self):
        start = vp.vector(*self.actor_1.phys_obj.pos.tolist())
        end = vp.vector(*self.actor_2.phys_obj.pos.tolist())
        axis = end - start

        self.visual.pos = start
        self.visual.axis = axis

def example_masses_on_spring():
    print("masses on a spring")
    
    scene = vp.canvas(title='spring')

    p1 = make_particle_actor('particle1',
                            pos=[2, 0, 0], lin_momentum=[0, 0, 0], mass=1.0)
    p2 = make_particle_actor('particle2',
                            pos=[-2, 0, 0], lin_momentum=[0, 0, 0], mass=1.0)

    spring = VisualLinkageParticles('spring',  p1, p2, visual=vp.helix(color = vp.color.green, radius=0.4, thickness=0.1),)

    forces = [
        HookeForce(p1.phys_obj, p2.phys_obj, ks=0.8, kd=0.0, rest_length=3.0)
    ]

    dt = 0.01
    sim_world = SimWorld(dt, 200, entities = [p1, p2, spring],
                        forces = forces,
                        dynamics_solver=EulerMethod())
    sim_world.run(n_steps=2000)

    print("FINISHED")
    print("Close tab in browser to exit")

def example_hanging_particle():
    print("hanging particle")
    scene = vp.canvas(title='Particle hanging on the spring')

    p1 = make_particle_actor('particle1',
                            pos=[2, 0, 0], lin_momentum=[0, 0, 0], mass=1.0)
    p1.phys_obj.inv_mass = 0 # make body with infinite mass


    p2 = make_particle_actor('particle2',
                            pos=[-2, 0, 0], lin_momentum=[0, 0, 0], mass=1.0)

    spring = VisualLinkageParticles('spring',  p1, p2, visual=vp.helix(color = vp.color.green, radius=0.4, thickness=0.1),)

    forces = [
        HookeForce(p1.phys_obj, p2.phys_obj, ks=2.8, kd=1.0, rest_length=3.0),
        GravityForce()
    ]
    dt = 0.01
    sim_world = SimWorld(dt, 200, entities = [p1, p2, spring],
                        forces = forces,
                        dynamics_solver=EulerMethod())
    sim_world.run(n_steps=2000)

    print("FINISHED")
    print("Close tab in browser to exit")
    
def example_wave():
    print("wave")
    scene = vp.canvas(title='Wave')

    N = 20
    l = 2.
        
    k_s = 1.5
    k_d = 0.11
    rest_length = l

    center_shift = Vec3((N - 1) * l / 2, 0, 0)

    helix_params = dict(color = vp.color.green, radius=0.2, thickness=0.1)
    forces = []
    entities = []
    linkages = []
    lin_momentum_scale = 0.1
    start_momentum = Vec3(1., 0, 0)

    for i in range(-1, N + 1):
        entities.append(make_particle_actor(f'particle_{i}',
                            pos=Vec3(l * i, 0, 0) - center_shift, lin_momentum=[0, 0, 0], mass=1.0, radius=0.5))
        if i == -1 or i == N:
            # entities[-1].visual.visible = False
            entities[-1].phys_obj.inv_mass = 0
            
        if i > -1:
            forces.append(HookeForce(entities[-1].phys_obj, entities[-2].phys_obj, ks=k_s, kd=k_d, rest_length=rest_length))
            linkages.append(VisualLinkageParticles(f'link_{i}', entities[-1], entities[-2], visual=vp.helix(**helix_params)))

        # if i == 1:
    entities[1].phys_obj.linear_momentum = start_momentum

    entities = entities + linkages

    dt=0.05
    sim_world = SimWorld(dt, 60, entities = entities,
                        forces = forces,
                        dynamics_solver=EulerMethod())
    sim_world.run(n_steps=2000)

    print("FINISHED")

    print("Close tab in browser to exit")

EXAMPLES = {
    'masses': example_masses_on_spring,
    'hanging': example_hanging_particle,
    'wave': example_wave,
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