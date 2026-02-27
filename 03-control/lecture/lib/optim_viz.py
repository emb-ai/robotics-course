from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from ipywidgets import interact, FloatSlider, Dropdown


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def _rosenbrock(x: NDArray) -> float:
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def _rosenbrock_grad(x: NDArray) -> NDArray:
    dx0 = -2 * (1 - x[0]) + 100 * 2 * (x[1] - x[0] ** 2) * (-2 * x[0])
    dx1 = 100 * 2 * (x[1] - x[0] ** 2)
    return np.array([dx0, dx1])


def _rosenbrock_hess(x: NDArray) -> NDArray:
    return np.array([
        [2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]],
        [-400 * x[0], 200.0],
    ])


def _quadratic(x: NDArray) -> float:
    H = np.array([[4.0, 1.0], [1.0, 2.0]])
    return float(0.5 * x @ H @ x)


def _quadratic_grad(x: NDArray) -> NDArray:
    H = np.array([[4.0, 1.0], [1.0, 2.0]])
    return H @ x


def _quadratic_hess(_x: NDArray) -> NDArray:
    return np.array([[4.0, 1.0], [1.0, 2.0]])


FUNCTIONS = {
    "rosenbrock": (_rosenbrock, _rosenbrock_grad, _rosenbrock_hess, np.array([1.0, 1.0])),
    "quadratic": (_quadratic, _quadratic_grad, _quadratic_hess, np.array([0.0, 0.0])),
}


def _feasible_polygon(A: NDArray, b: NDArray) -> NDArray | None:
    n = A.shape[0]
    vertices = []
    for i in range(n):
        for j in range(i + 1, n):
            A_pair = A[[i, j]]
            b_pair = b[[i, j]]
            try:
                pt = np.linalg.solve(A_pair, b_pair)
            except np.linalg.LinAlgError:
                continue
            if np.all(A @ pt <= b + 1e-9):
                vertices.append(pt)
    if len(vertices) < 3:
        return None
    vertices = np.array(vertices)
    hull = ConvexHull(vertices)
    return vertices[hull.vertices]


# ---------------------------------------------------------------------------
# Gradient descent / Newton path utilities
# ---------------------------------------------------------------------------

def _gradient_descent_path(
    x0: NDArray, grad_fn, lr: float, n_steps: int,
) -> NDArray:
    path = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        x = x - lr * grad_fn(x)
        path.append(x.copy())
    return np.array(path)


def _newton_path(
    x0: NDArray, grad_fn, hess_fn, n_steps: int,
) -> NDArray:
    path = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        H = hess_fn(x)
        g = grad_fn(x)
        try:
            dx = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        x = x - dx
        path.append(x.copy())
    return np.array(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def show_unconstrained_optimization():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (name, (fn, grad_fn, hess_fn, x_opt)) in zip(axes, FUNCTIONS.items()):
        if name == "rosenbrock":
            xr = np.linspace(-2, 2, 300)
            yr = np.linspace(-1, 3, 300)
            x0 = np.array([-1.5, 2.0])
            lr_gd = 0.001
            n_gd, n_nt = 500, 30
        else:
            xr = np.linspace(-3, 3, 300)
            yr = np.linspace(-3, 3, 300)
            x0 = np.array([2.5, 2.5])
            lr_gd = 0.15
            n_gd, n_nt = 30, 5

        X, Y = np.meshgrid(xr, yr)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = fn(np.array([X[i, j], Y[i, j]]))

        levels = np.logspace(np.log10(max(Z.min(), 1e-3)), np.log10(Z.max()), 30) if name == "rosenbrock" else 20
        ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.7)

        gd_path = _gradient_descent_path(x0, grad_fn, lr_gd, n_gd)
        nt_path = _newton_path(x0, grad_fn, hess_fn, n_nt)

        ax.plot(gd_path[:, 0], gd_path[:, 1], "o-", color="blue", markersize=2, lw=1, label="Gradient descent")
        ax.plot(nt_path[:, 0], nt_path[:, 1], "s-", color="red", markersize=3, lw=1.2, label="Newton's method")
        ax.plot(*x_opt, "k*", markersize=14, zorder=10, label="Optimum")
        ax.plot(*x0, "go", markersize=8, zorder=10, label="Start")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(name.capitalize())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def show_constrained_optimization():
    H = np.array([[2.0, 0.0], [0.0, 2.0]])
    c_vec = np.array([-2.0, -5.0])

    def objective(x):
        return 0.5 * x @ H @ x + c_vec @ x

    def obj_grad(x):
        return H @ x + c_vec

    # constraints: x0 >= 0, x1 >= 0, x0 + x1 <= 4, -x0 + 2*x1 <= 6
    A_ub = np.array([[-1, 0], [0, -1], [1, 1], [-1, 2]])
    b_ub = np.array([0, 0, 4, 6])

    constraints = [{"type": "ineq", "fun": lambda x, i=i: b_ub[i] - A_ub[i] @ x} for i in range(len(b_ub))]
    res = minimize(objective, np.array([0.5, 0.5]), method="SLSQP", constraints=constraints)
    x_star = res.x

    fig, ax = plt.subplots(figsize=(8, 7))

    xr = np.linspace(-0.5, 5, 300)
    yr = np.linspace(-0.5, 5, 300)
    X, Y = np.meshgrid(xr, yr)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective(np.array([X[i, j], Y[i, j]]))

    ax.contour(X, Y, Z, levels=20, cmap="coolwarm", alpha=0.7)

    verts = _feasible_polygon(A_ub, b_ub)
    if verts is not None:
        poly = plt.Polygon(verts, alpha=0.15, color="green", label="Feasible region")
        ax.add_patch(poly)
        ax.plot(np.append(verts[:, 0], verts[0, 0]),
                np.append(verts[:, 1], verts[0, 1]),
                "g-", lw=1.5)

    ax.plot(*x_star, "r*", markersize=16, zorder=10, label=f"Optimum ({x_star[0]:.2f}, {x_star[1]:.2f})")

    # KKT: show gradient of objective and active constraint normals
    grad_f = obj_grad(x_star)
    scale = 0.8
    ax.annotate("", xy=x_star - scale * grad_f, xytext=x_star,
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(x_star[0] - scale * grad_f[0], x_star[1] - scale * grad_f[1] - 0.2,
            r"$-\nabla f$", color="red", fontsize=10)

    for i in range(len(b_ub)):
        slack = b_ub[i] - A_ub[i] @ x_star
        if abs(slack) < 0.05:
            normal = -A_ub[i]
            normal_unit = normal / (np.linalg.norm(normal) + 1e-12)
            ax.annotate("", xy=x_star + scale * normal_unit, xytext=x_star,
                        arrowprops=dict(arrowstyle="->", color="blue", lw=2))

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Constrained optimization with KKT conditions")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def show_qp_example():
    H = np.array([[2.0, 0.5], [0.5, 1.0]])
    c_vec = np.array([-1.0, -2.0])

    A_ub = np.array([[1, 1], [-1, 0], [0, -1], [2, -1]])
    b_ub = np.array([3.0, 0.0, 0.0, 4.0])

    def objective(x):
        return 0.5 * x @ H @ x + c_vec @ x

    constraints = [{"type": "ineq", "fun": lambda x, i=i: b_ub[i] - A_ub[i] @ x} for i in range(len(b_ub))]
    res = minimize(objective, np.array([0.5, 0.5]), method="SLSQP", constraints=constraints)
    x_star = res.x

    fig, ax = plt.subplots(figsize=(8, 7))

    xr = np.linspace(-0.5, 4, 300)
    yr = np.linspace(-0.5, 4, 300)
    X, Y = np.meshgrid(xr, yr)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective(np.array([X[i, j], Y[i, j]]))

    ax.contour(X, Y, Z, levels=20, cmap="coolwarm", alpha=0.7)

    verts = _feasible_polygon(A_ub, b_ub)
    if verts is not None:
        poly = plt.Polygon(verts, alpha=0.15, color="green", label="Feasible region")
        ax.add_patch(poly)
        ax.plot(np.append(verts[:, 0], verts[0, 0]),
                np.append(verts[:, 1], verts[0, 1]),
                "g-", lw=1.5)

    ax.plot(*x_star, "r*", markersize=16, zorder=10,
            label=f"QP solution ({x_star[0]:.3f}, {x_star[1]:.3f})")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(r"QP: $\min\;\frac{1}{2}x^\top H x + c^\top x$ s.t. $Ax \leq b$")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    print(f"Optimal x = [{x_star[0]:.4f}, {x_star[1]:.4f}]")
    print(f"Optimal value = {res.fun:.4f}")


def show_gradient_descent_interactive():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _update(function: str = "quadratic", step_size: float = 0.01, n_steps: int = 50):
        fn, grad_fn, _, x_opt = FUNCTIONS[function]

        if function == "rosenbrock":
            x0 = np.array([-1.5, 2.0])
            xr = np.linspace(-2, 2, 300)
            yr = np.linspace(-1, 3, 300)
        else:
            x0 = np.array([2.5, 2.5])
            xr = np.linspace(-3, 3, 300)
            yr = np.linspace(-3, 3, 300)

        X, Y = np.meshgrid(xr, yr)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = fn(np.array([X[i, j], Y[i, j]]))

        path = _gradient_descent_path(x0, grad_fn, step_size, n_steps)
        values = np.array([fn(xi) for xi in path])

        ax = axes[0]
        ax.clear()
        levels = np.logspace(np.log10(max(Z.min(), 1e-3)), np.log10(Z.max()), 30) if function == "rosenbrock" else 20
        ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.7)
        ax.plot(path[:, 0], path[:, 1], "o-", color="blue", markersize=3, lw=1)
        ax.plot(*x_opt, "k*", markersize=14, zorder=10)
        ax.plot(*x0, "go", markersize=8, zorder=10)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(f"GD on {function} (lr={step_size:.4f}, {n_steps} steps)")
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.clear()
        ax2.semilogy(values, "b.-", lw=1)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("$f(x)$")
        ax2.set_title("Objective value")
        ax2.grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    interact(
        _update,
        function=Dropdown(options=list(FUNCTIONS.keys()), value="quadratic", description="Function"),
        step_size=FloatSlider(min=0.0001, max=0.5, step=0.0001, value=0.01,
                              description="Step size", readout_format=".4f"),
        n_steps=FloatSlider(min=5, max=500, step=5, value=50, description="Steps"),
    )
    plt.show()
