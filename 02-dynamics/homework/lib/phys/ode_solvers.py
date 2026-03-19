from abc import abstractmethod, ABC
from typing import Callable
import numpy as np


class ODESolverBase(ABC):
    @abstractmethod
    def ode(self, x_0, t_0, t_1, dx_dt_func: Callable):
        raise NotImplementedError


class EulerMethod(ODESolverBase):
    '''
    x_0 = (x_0, v_0)
    dx_dt_func = (f(x, v), g(x, v))

    Step:
    x_1 = x_0 + dt * f(x_0, v_0)
    v_1 = v_0 + dt * g(x_0, v_0)
    '''
    def ode(self, x_0, t_0, t_1, dx_dt_func):
        dt = t_1 - t_0
        f = dx_dt_func(x_0)
        x_1 = x_0['x'] + dt * f['x_dot']
        p_1 = x_0['p'] + dt * f['p_dot']
        return {
            'x': x_1,
            'p': p_1
        }


class SemiImplicitEulerMethod(ODESolverBase):
    '''
    x_0 = (x_0, v_0)
    dx_dt_func = (f(x, v), g(x, v))

    Step:
    v_1 = v_0 + dt * g(x_0, v_0)
    x_1 = x_0 + dt * f(x_0, v_1)
    '''
    def ode(self, x_0, t_0, t_1, dx_dt_func):
        dt = t_1 - t_0

        # first update velocity
        p_1 = x_0['p'] + dt * dx_dt_func(x_0)['p_dot']

        # then update x using new velocity
        x_1 = x_0['x'] + dt * dx_dt_func(  # recalculate with new velocity
            {
                'x': x_0['x'],
                'p': p_1
            }
        )['x_dot']

        return {'x': x_1,
            'p': p_1
        }
