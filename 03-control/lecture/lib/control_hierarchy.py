from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


LAYERS = [
    {
        "name": "Task / Mission Planning",
        "rate": "~0.1–1 Hz",
        "methods": "FSM, Behavior Trees, Task Planners",
        "color": "#4C72B0",
    },
    {
        "name": "Motion Planning",
        "rate": "~1–10 Hz",
        "methods": "RRT, A*, Trajectory Optimization",
        "color": "#55A868",
    },
    {
        "name": "Trajectory Tracking / MPC",
        "rate": "~10–100 Hz",
        "methods": "MPC, LQR, Feedback Linearization",
        "color": "#C44E52",
    },
    {
        "name": "Low-level Servo / PID",
        "rate": "~1–10 kHz",
        "methods": "PID, Current/Torque Control",
        "color": "#8172B2",
    },
]

METHOD_TABLE = [
    ("PID",      "Low-level Servo",       True,  "1–10 kHz"),
    ("LQR",      "Trajectory Tracking",   True,  "10–100 Hz"),
    ("MPC",      "Trajectory Tracking",   True,  "10–100 Hz"),
    ("Lyapunov", "Trajectory Tracking",   True,  "10–100 Hz"),
    ("IK",       "Motion Planning",       True,  "1–10 Hz"),
    ("RL",       "Task / Motion Planning", False, "1–100 Hz"),
]


def show_control_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(LAYERS) * 2.2 + 0.5)
    ax.axis("off")
    ax.set_title("Control Hierarchy in Modern Robots", fontsize=14, fontweight="bold", pad=12)

    box_w, box_h = 8.0, 1.5
    x0 = (10 - box_w) / 2
    gap = 0.6

    for i, layer in enumerate(LAYERS):
        y = (len(LAYERS) - 1 - i) * (box_h + gap)
        rect = mpatches.FancyBboxPatch(
            (x0, y), box_w, box_h,
            boxstyle="round,pad=0.15",
            facecolor=layer["color"], edgecolor="white", linewidth=2, alpha=0.88,
        )
        ax.add_patch(rect)
        cx = x0 + box_w / 2
        cy = y + box_h / 2
        ax.text(cx, cy + 0.3, layer["name"],
                ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        ax.text(cx, cy - 0.15, layer["rate"],
                ha="center", va="center", fontsize=9, color="white", alpha=0.9)
        ax.text(cx, cy - 0.5, layer["methods"],
                ha="center", va="center", fontsize=8, color="white", alpha=0.8, style="italic")

        if i < len(LAYERS) - 1:
            arrow_y_top = y
            arrow_y_bot = y - gap
            arrow = FancyArrowPatch(
                (cx, arrow_y_top - 0.05), (cx, arrow_y_bot + 0.05),
                arrowstyle="-|>", mutation_scale=18,
                color="#444444", linewidth=1.8,
            )
            ax.add_patch(arrow)

    fig.tight_layout()
    plt.show()


def show_method_hierarchy_table() -> None:
    header = f"{'Method':<10} {'Layer':<24} {'Model-based':<14} {'Typical Rate'}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for method, layer, model_based, rate in METHOD_TABLE:
        mb = "Yes" if model_based else "No"
        print(f"{method:<10} {layer:<24} {mb:<14} {rate}")
    print(sep)
