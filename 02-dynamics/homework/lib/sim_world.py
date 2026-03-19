from typing import Optional

from lib.phys.physics_world import PhysicsEngine
from lib.entity import Actor


class SimWorld:
    def __init__(self, dt, rate, entities, **phys_kwargs):
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
                    raise ValueError(f"Duplicate phys object name: {entity.phys_obj.name}")
                self.phys_obj_names.append(entity.phys_obj.name)
            if entity.name in self.entity_names:
                raise ValueError(f"Duplicate entity name: {entity.name}")
            self.entity_names.append(entity.name)

        self.phys_world = PhysicsEngine(
                    bodies = phys_bodies,
                    **phys_kwargs
                )

    
    def step(self, t_0, t_1):
        if len(self.entities) > 0:
            self.phys_world.step(t_0, t_1)
        self.update_visuals()

    def update_visuals(self):
        for entities in self.entities:
            entities.update_visuals()


    def run(self, n_steps: Optional[int] = None):
        import vpython as vp
        elapsed_steps = 0
        cut_t = 0
        while True:
            vp.rate(self.rate)
            self.step(cut_t, cut_t + self.dt)
            cut_t += self.dt
            elapsed_steps += 1
            if n_steps is not None and elapsed_steps >= n_steps:
                break
