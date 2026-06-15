"""Small plotting helpers for the student version."""
from __future__ import annotations
import math
import matplotlib.pyplot as plt
import torch
from .robot_model import COLLISION_LINK_NAMES

def _to_numpy(x):
    return x.detach().cpu().numpy()

def _set_equal_3d(ax, points):
    pts = _to_numpy(points.reshape(-1, 3))
    center = pts.mean(axis=0)
    span = max((pts.max(axis=0) - pts.min(axis=0)).max(), 0.1)
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - span * 0.6, c + span * 0.6)

def _tcp_positions(planner, q_traj, inputs):
    batch, steps = q_traj.shape[:2]
    q_flat = q_traj.reshape(batch * steps, 7)
    T = planner.tcp_pose(q_flat, inputs)
    return T[:, :3, 3].reshape(batch, steps, 3)

def _link_points(planner, q_traj, inputs):
    batch, steps = q_traj.shape[:2]
    q_flat = q_traj.reshape(batch * steps, 7)
    pts = planner.link_control_points_world(q_flat, inputs)
    return pts.reshape(batch, steps, pts.shape[1], pts.shape[2], 3)

def _plot_sphere(ax, center, radius, alpha=0.18):
    import numpy as np
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0, alpha=alpha)

def _obstacles(inputs):
    centers = _to_numpy(inputs['obstacle_centers'])[0]
    radii = _to_numpy(inputs['obstacle_radii'])[0]
    safety = float(_to_numpy(inputs['collision_safety_margin']).reshape(-1)[0])
    link_radius = float(_to_numpy(inputs['link_collision_radius']).reshape(-1)[0])
    return (centers, radii, safety, link_radius)

def _margins(planner, q_traj, inputs):
    centers = inputs['obstacle_centers']
    radii = inputs['obstacle_radii']
    safety = inputs['collision_safety_margin'].reshape(-1, 1, 1, 1)
    link_r = inputs['link_collision_radius'].reshape(-1, 1, 1, 1)
    points = _link_points(planner, q_traj, inputs)
    dist = torch.linalg.norm(points[:, :, None, :, :, :] - centers[:, None, :, None, None, :], dim=-1)
    required = radii[:, None, :, None, None] + safety + link_r
    return dist - required

def plot_tcp_trajectory_3d(planner, q_init, q_solution, tcp_goal_pos, inputs, save_path='tcp_trajectory_before_after.png', method_label='trajectory'):
    """Plot TCP trajectory before and after Theseus for one initialization method.

    method_label examples:
        "linear warm-start"
        "MLP warm-start"
    """
    before = _tcp_positions(planner, q_init, inputs)[0]
    after = _tcp_positions(planner, q_solution, inputs)[0]
    goal = tcp_goal_pos[0]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*_to_numpy(before).T, linestyle='--', label=f'{method_label}: before Theseus')
    ax.plot(*_to_numpy(after).T, label=f'{method_label}: after Theseus')
    ax.scatter(*_to_numpy(goal), marker='*', s=120, label='TCP goal')
    centers, radii, safety, _ = _obstacles(inputs)
    for c, r in zip(centers, radii):
        _plot_sphere(ax, c, r + safety, alpha=0.15)
    ax.set_title(f'TCP trajectory ({method_label})')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    _set_equal_3d(ax, torch.cat([before, after, goal.view(1, 3)], dim=0))
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    pass

def plot_tcp_trajectory_linear_vs_mlp_3d(planner, q_linear_init, q_linear_solution, q_mlp_init, q_mlp_solution, tcp_goal_pos, inputs_linear, inputs_mlp, save_path='tcp_trajectory_full_planner_linear_vs_mlp.png'):
    """Plot linear and MLP TCP trajectories before/after Theseus on one 3D plot."""
    linear_before = _tcp_positions(planner, q_linear_init, inputs_linear)[0]
    linear_after = _tcp_positions(planner, q_linear_solution, inputs_linear)[0]
    mlp_before = _tcp_positions(planner, q_mlp_init, inputs_mlp)[0]
    mlp_after = _tcp_positions(planner, q_mlp_solution, inputs_mlp)[0]
    goal = tcp_goal_pos[0]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*_to_numpy(linear_before).T, linestyle='--', label='linear: before Theseus')
    ax.plot(*_to_numpy(linear_after).T, label='linear: after Theseus')
    ax.plot(*_to_numpy(mlp_before).T, linestyle='--', label='MLP: before Theseus')
    ax.plot(*_to_numpy(mlp_after).T, label='MLP: after Theseus')
    ax.scatter(*_to_numpy(goal), marker='*', s=120, label='TCP goal')
    centers, radii, safety, _ = _obstacles(inputs_linear)
    for c, r in zip(centers, radii):
        _plot_sphere(ax, c, r + safety, alpha=0.15)
    ax.set_title('TCP trajectory comparison: linear and MLP warm-starts')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    all_points = torch.cat([linear_before, linear_after, mlp_before, mlp_after, goal.view(1, 3)], dim=0)
    _set_equal_3d(ax, all_points)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    pass

def plot_robot_start_goal_points(planner, q_init, q_solution, inputs, save_path='link_control_point_snapshots.png'):
    """Plot robot link collision control points only at the initial configuration.

    q_solution is kept in the function signature for backward compatibility with
    existing calls, but it is intentionally not plotted.
    """
    start = _link_points(planner, q_init[:, :1], inputs)[0, 0].reshape(-1, 3)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*_to_numpy(start).T, s=8, label='robot collision points at start')
    centers, radii, safety, link_radius = _obstacles(inputs)
    for c, r in zip(centers, radii):
        _plot_sphere(ax, c, r, alpha=0.2)
        _plot_sphere(ax, c, r + safety + link_radius, alpha=0.08)
    ax.set_title('Robot collision points at start')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    _set_equal_3d(ax, start)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    pass

def plot_clearance_over_time(planner, q_init, q_solution, inputs, save_path='link_clearance_before_after.png', method_label='trajectory'):
    before = _margins(planner, q_init, inputs).amin(dim=(2, 3, 4))[0]
    after = _margins(planner, q_solution, inputs).amin(dim=(2, 3, 4))[0]
    x = range(before.numel())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, _to_numpy(before), label=f'{method_label}: before Theseus')
    ax.plot(x, _to_numpy(after), label=f'{method_label}: after Theseus')
    ax.axhline(0.0, linestyle='--', linewidth=1)
    ax.set_title(f'Minimum obstacle clearance over trajectory ({method_label})')
    ax.set_xlabel('Trajectory knot')
    ax.set_ylabel('Clearance [m]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    pass

def plot_clearance_per_link_over_time(planner, q_init, q_solution, inputs, save_path='link_clearance_per_link_before_after.png', method_label='trajectory'):
    """Plot minimum clearance for each collision link before/after optimization."""
    before = _margins(planner, q_init, inputs)[0].amin(dim=(1, 3))
    after = _margins(planner, q_solution, inputs)[0].amin(dim=(1, 3))
    num_links = before.shape[1]
    steps = range(before.shape[0])
    cols = 2
    rows = math.ceil(num_links / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.4 * rows), sharex=True)
    axes = axes.reshape(-1)
    for link_idx in range(num_links):
        ax = axes[link_idx]
        name = COLLISION_LINK_NAMES[link_idx] if link_idx < len(COLLISION_LINK_NAMES) else f'link {link_idx}'
        ax.plot(steps, _to_numpy(before[:, link_idx]), linestyle='--', label=f'{method_label}: before Theseus')
        ax.plot(steps, _to_numpy(after[:, link_idx]), label=f'{method_label}: after Theseus')
        ax.axhline(0.0, linestyle='--', linewidth=1)
        ax.set_title(name)
        ax.set_ylabel('Clearance [m]')
        ax.grid(True, alpha=0.3)
        if link_idx == 0:
            ax.legend(fontsize=8)
    for ax in axes[num_links:]:
        ax.axis('off')
    for ax in axes[-cols:]:
        if ax.has_data():
            ax.set_xlabel('Trajectory knot')
    fig.suptitle(f'Minimum obstacle clearance per robot link ({method_label})')
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    pass

def plot_joint_trajectories(q_init, q_solution, save_path='joint_trajectories_before_after.png', method_label='trajectory'):
    """Plot every joint before/after optimization for one warm-start method."""
    qi = _to_numpy(q_init[0])
    qs = _to_numpy(q_solution[0])
    steps = range(qi.shape[0])
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True)
    axes = axes.reshape(-1)
    for j in range(7):
        ax = axes[j]
        ax.plot(steps, qi[:, j], linestyle='--', label=f'{method_label}: before Theseus')
        ax.plot(steps, qs[:, j], label=f'{method_label}: after Theseus')
        ax.set_title(f'Joint q{j + 1}')
        ax.set_ylabel('Angle [rad]')
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    axes[7].axis('off')
    for ax in axes[4:7]:
        ax.set_xlabel('Trajectory knot')
    fig.suptitle(f'Joint trajectories before and after optimization ({method_label})')
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    pass

def plot_tcp_error_over_time(planner, q_init, q_solution, tcp_goal_pos, inputs, save_path='tcp_error_linear_vs_mlp_before_after.png', method_label='trajectory'):
    before = torch.linalg.norm(_tcp_positions(planner, q_init, inputs)[0] - tcp_goal_pos[0], dim=1)
    after = torch.linalg.norm(_tcp_positions(planner, q_solution, inputs)[0] - tcp_goal_pos[0], dim=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(_to_numpy(before), linestyle='--', label=f'{method_label}: before Theseus')
    ax.plot(_to_numpy(after), label=f'{method_label}: after Theseus')
    ax.set_title(f'TCP position error to goal ({method_label})')
    ax.set_xlabel('Trajectory knot')
    ax.set_ylabel('Error [m]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    pass

def plot_tcp_error_linear_vs_mlp_over_time(planner, q_linear_init, q_linear_solution, q_mlp_init, q_mlp_solution, tcp_goal_pos, inputs_linear, inputs_mlp, save_path='tcp_error_linear_vs_mlp_before_after.png'):
    """Plot TCP error for linear/MLP before and after Theseus on one figure."""
    linear_before = torch.linalg.norm(_tcp_positions(planner, q_linear_init, inputs_linear)[0] - tcp_goal_pos[0], dim=1)
    linear_after = torch.linalg.norm(_tcp_positions(planner, q_linear_solution, inputs_linear)[0] - tcp_goal_pos[0], dim=1)
    mlp_before = torch.linalg.norm(_tcp_positions(planner, q_mlp_init, inputs_mlp)[0] - tcp_goal_pos[0], dim=1)
    mlp_after = torch.linalg.norm(_tcp_positions(planner, q_mlp_solution, inputs_mlp)[0] - tcp_goal_pos[0], dim=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(_to_numpy(linear_before), linestyle='--', label='linear: before Theseus')
    ax.plot(_to_numpy(mlp_before), linestyle='--', label='MLP: before Theseus')
    ax.plot(_to_numpy(linear_after), label='linear: after Theseus')
    ax.plot(_to_numpy(mlp_after), label='MLP: after Theseus')
    ax.set_title('TCP position error comparison: linear and MLP warm-starts')
    ax.set_xlabel('Trajectory knot')
    ax.set_ylabel('Error [m]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    pass
