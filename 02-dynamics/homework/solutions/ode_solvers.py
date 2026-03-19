"""Student solution: RK4 and re-exports from lib. Task 1.2."""
from typing import Callable

from lib.phys.ode_solvers import (
    EulerMethod,
    ODESolverBase,
    SemiImplicitEulerMethod,
)


class RK4Method(ODESolverBase):
    """Classic 4th-order Runge-Kutta. Implement in solutions."""

    def ode(self, x_0, t_0, t_1, dx_dt_func: Callable):
        # YOUR CODE HERE: one RK4 step, return {"x": x_1, "p": p_1}
        raise NotImplementedError("Implement RK4 step.")


__all__ = ["EulerMethod", "SemiImplicitEulerMethod", "ODESolverBase", "RK4Method"]
