"""Student solution: impulse-based constraint manager. Task 2.1."""
from dataclasses import dataclass

from lib.phys.constraints.manager import ConstraintsManager


@dataclass
class ConstraintsManagerImpulseBased(ConstraintsManager):
    beta_baumgarte: float = 0.5

    def calc_forces(self, t_0, t_1):
        if len(self.constraints) == 0:
            return None
        # YOUR CODE HERE: update constraints, build J/W, solve for lambda, return F_C
        raise NotImplementedError("Implement impulse-based constraint forces.")
