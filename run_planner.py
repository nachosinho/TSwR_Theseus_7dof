"""Prosty skrypt do uruchomienia planera Panda 7DoF.

Domyślnie odpala zwykły liniowy warm-start. Można też użyć MLP albo
porównać oba warianty przez --warmstart-mode both. Pliki wynikowe są
zapisywane w stałych lokalizacjach: modele i trajektorie obok skryptów, wykresy oraz CSV w ./data/.
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from franka_dgpmp2.config import LINK_COLLISION_RADIUS, MAX_ITERATIONS, NUM_STEPS, DEFAULT_SCENARIO_NAME, POINTS_PER_LINK, SAFETY_MARGIN, TCP_LINK_NAME, PlannerWeights, SCENARIOS, get_scenario
from franka_dgpmp2.kinematics import fk_urdf_tcp, rpy_to_matrix, tool_transform_for_tcp_link
from franka_dgpmp2.planner import Theseus7DofPlannerMVP
from franka_dgpmp2.plotting import plot_clearance_over_time, plot_clearance_per_link_over_time, plot_joint_trajectories, plot_robot_start_goal_points, plot_tcp_error_over_time, plot_tcp_trajectory_3d
from franka_dgpmp2.robot_model import COLLISION_LINK_NAMES, RobotURDFKinematics, default_link_control_points
from franka_dgpmp2.warmstart_mlp import predict_warmstart_q_init
DEFAULT_SUCCESS_TCP_THRESHOLD_M = 0.005
DEFAULT_SUCCESS_MARGIN_THRESHOLD_M = 0.0
DEFAULT_MANUAL_TARGET_IK_STEPS = 500
DEFAULT_MANUAL_TARGET_IK_LR = 0.04
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data/figures'

def t(x, dtype, device):
    return torch.tensor(x, dtype=dtype, device=device)

def root_file(filename: str | Path) -> Path:
    """File generated next to the runnable scripts: models and trajectories."""
    return SCRIPT_DIR / Path(filename).name

def data_file(filename: str | Path) -> Path:
    """File generated in ./data/: plots and CSV tables."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / Path(filename).name

def has_waypoint_targets(args) -> bool:
    """Return True when the CLI contains at least one manual waypoint target."""
    return bool(getattr(args, 'waypoint_pos', None) or getattr(args, 'waypoints_file', None))

def has_manual_position_target(args) -> bool:
    """Return True when any manual TCP position target is configured."""
    return bool(getattr(args, 'target_pos', None) is not None or has_waypoint_targets(args))

def ignore_manual_goal_rotation(args, target_pos=None) -> bool:
    """Return True when a manually supplied TCP position should be position-only."""
    manual_position_active = target_pos is not None or has_manual_position_target(args)
    return bool(manual_position_active and args.target_rpy is None and (not args.enforce_target_rotation))

def make_planner(args, scenario, dtype, device):
    weights = PlannerWeights()
    if ignore_manual_goal_rotation(args):
        weights = replace(weights, tcp_rot=0.0)
    return Theseus7DofPlannerMVP(num_steps=NUM_STEPS, weights=weights, max_iterations=args.max_iterations, step_size=0.2, dtype=dtype, device=device, link_control_points_per_link=POINTS_PER_LINK, num_obstacles=len(scenario.obstacle_centers))

def tcp_pose_from_q(q, robot, tool):
    return fk_urdf_tcp(q, robot.joint_origin_xyz.view(1, 7, 3), robot.joint_origin_rpy.view(1, 7, 3), robot.joint_axis.view(1, 7, 3), robot.tcp_fixed_transform.view(1, 4, 4), tool)

def solve_manual_target_ik(robot, tool, q_seed, tcp_goal_pos, tcp_goal_rot=None, *, steps=DEFAULT_MANUAL_TARGET_IK_STEPS, lr=DEFAULT_MANUAL_TARGET_IK_LR):
    """Find a joint-space endpoint used only as warm-start for a manual TCP goal.

    Theseus/GPMP2 still optimizes the final trajectory. This small torch-based
    IK pass only gives the linear/MLP warm-start a q_target consistent with the
    manually supplied TCP point.
    """
    steps = max(1, int(steps))
    q_seed = q_seed.detach()
    q = q_seed.clone().detach().requires_grad_(True)
    joint_min = robot.joint_min.view(1, 7)
    joint_max = robot.joint_max.view(1, 7)
    opt = torch.optim.Adam([q], lr=float(lr))
    best_q = q.detach().clone()
    best_score = float('inf')
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        T = tcp_pose_from_q(q, robot, tool)
        pos_err = T[:, :3, 3] - tcp_goal_pos
        pos_loss = torch.sum(pos_err ** 2)
        reg_loss = torch.sum((q - q_seed) ** 2)
        loss = 1000.0 * pos_loss + 0.01 * reg_loss
        if tcp_goal_rot is not None:
            rot_loss = torch.sum((T[:, :3, :3] - tcp_goal_rot) ** 2)
            loss = loss + 0.5 * rot_loss
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.clamp_(joint_min, joint_max)
            score = float(pos_loss.detach().cpu())
            if score < best_score:
                best_score = score
                best_q = q.detach().clone()
    with torch.no_grad():
        T_best = tcp_pose_from_q(best_q, robot, tool)
        pos_error_m = float(torch.linalg.norm(T_best[:, :3, 3] - tcp_goal_pos, dim=1).cpu()[0])
        rot_error_deg = None
        if tcp_goal_rot is not None:
            rot_error_deg = float(rotation_error_deg(tcp_goal_rot, T_best[:, :3, :3]).cpu()[0])
    return (best_q, {'pos_error_m': pos_error_m, 'rot_error_deg': rot_error_deg})

def make_problem(planner, scenario, args, dtype, device, *, target_pos_override=None, q_start_override=None, segment_name=None):
    robot = RobotURDFKinematics.franka_panda(dtype=dtype, device=device)
    tool = tool_transform_for_tcp_link(TCP_LINK_NAME, 1, dtype=dtype, device=device)
    if q_start_override is None:
        q_start = t([scenario.q_start], dtype, device)
    else:
        q_start = torch.as_tensor(q_start_override, dtype=dtype, device=device).reshape(1, 7)
    q_target = t([scenario.q_target], dtype, device)
    scenario_goal_pose = tcp_pose_from_q(q_target, robot, tool)
    tcp_goal_pos = scenario_goal_pose[:, :3, 3]
    tcp_goal_rot = scenario_goal_pose[:, :3, :3]
    goal_source = 'scenario'
    goal_rotation_active = True
    ik_info = None
    target_pos = target_pos_override if target_pos_override is not None else args.target_pos
    if target_pos is not None:
        tcp_goal_pos = t([target_pos], dtype, device)
        goal_source = 'manual_tcp_position'
    if args.target_rpy is not None:
        tcp_goal_rot = rpy_to_matrix(t([args.target_rpy], dtype, device))
        goal_source = goal_source + '+manual_tcp_rpy'
    if ignore_manual_goal_rotation(args, target_pos):
        goal_rotation_active = False
    if target_pos is not None:
        ik_rot = tcp_goal_rot if goal_rotation_active else None
        ik_seed = q_start if q_start_override is not None else q_target
        q_target, ik_info = solve_manual_target_ik(robot, tool, ik_seed, tcp_goal_pos, ik_rot)
        if not goal_rotation_active:
            tcp_goal_rot = tcp_pose_from_q(q_target, robot, tool)[:, :3, :3]
    q_init = planner.linear_interpolation(q_start, q_target, NUM_STEPS)
    problem = {'robot': robot, 'tool_transform': tool, 'q_start': q_start, 'q_target': q_target, 'q_init': q_init, 'tcp_goal_pos': tcp_goal_pos, 'tcp_goal_rot': tcp_goal_rot, 'goal_source': goal_source, 'goal_rotation_active': goal_rotation_active, 'manual_target_ik': ik_info, 'link_points': default_link_control_points(points_per_link=POINTS_PER_LINK, dtype=dtype, device=device), 'obstacle_centers': scenario.obstacle_centers, 'obstacle_radii': scenario.obstacle_radii, 'scenario_name': scenario.name, 'segment_name': segment_name}
    build_inputs(planner, problem)
    return problem

def build_inputs(planner, problem):
    q_init = problem['q_init']
    dtype, device = (q_init.dtype, q_init.device)
    problem['inputs'] = planner.make_inputs(q_start=problem['q_start'], q_init=q_init, tcp_goal_pos=problem['tcp_goal_pos'], tcp_goal_rot=problem['tcp_goal_rot'], robot_model=problem['robot'], tool_transform=problem['tool_transform'], obstacle_centers=t([problem['obstacle_centers']], dtype, device), obstacle_radii=t([problem['obstacle_radii']], dtype, device), collision_safety_margin=t([[SAFETY_MARGIN]], dtype, device), link_collision_radius=t([[LINK_COLLISION_RADIUS]], dtype, device), link_control_points=problem['link_points'])

def copy_problem(planner, problem, new_q_init):
    copied = dict(problem)
    copied['q_init'] = new_q_init
    build_inputs(planner, copied)
    return copied

def load_mlp_warmstart(args, planner, problem, dtype, device):
    ckpt = Path(args.mlp_checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f'Nie znaleziono checkpointa MLP: {ckpt}')
    q_mlp, delta, info = predict_warmstart_q_init(q_init_linear=problem['q_init'], q_start=problem['q_start'], q_target=problem['q_target'], obstacle_centers=problem['obstacle_centers'], obstacle_radii=problem['obstacle_radii'], checkpoint_path=ckpt, target_num_steps=NUM_STEPS, dtype=dtype, device=device)
    pass
    pass
    pass
    pass
    return copy_problem(planner, problem, q_mlp)

def rotation_error_deg(R_goal, R):
    R_err = R_goal.transpose(1, 2) @ R
    cos_a = ((R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2] - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cos_a))

def link_margin(planner, q_traj, inputs):
    q_flat = q_traj.reshape(-1, 7)
    points = planner.link_control_points_world(q_flat, inputs)
    batch, steps = q_traj.shape[:2]
    points = points.reshape(batch, steps, points.shape[1], points.shape[2], 3)
    centers = inputs['obstacle_centers']
    radii = inputs['obstacle_radii']
    safety = inputs['collision_safety_margin'].reshape(-1, 1, 1, 1)
    link_r = inputs['link_collision_radius'].reshape(-1, 1, 1, 1)
    dist = torch.linalg.norm(points[:, :, None] - centers[:, None, :, None, None], dim=-1)
    required_dist = radii[:, None, :, None, None] + safety + link_r
    return dist - required_dist

def min_clearance(planner, q_traj, inputs):
    return link_margin(planner, q_traj, inputs).amin(dim=(1, 2, 3, 4))

def get_metrics(planner, problem, q_sol, inputs, solve_time, *, success_tcp_threshold_m=DEFAULT_SUCCESS_TCP_THRESHOLD_M, success_margin_threshold_m=DEFAULT_SUCCESS_MARGIN_THRESHOLD_M):
    T = planner.tcp_pose(q_sol[:, -1, :], inputs)
    pos_err = torch.linalg.norm(T[:, :3, 3] - problem['tcp_goal_pos'], dim=1)
    rot_err = rotation_error_deg(problem['tcp_goal_rot'], T[:, :3, :3])
    max_step = torch.linalg.norm(q_sol[:, 1:] - q_sol[:, :-1], dim=2).max()
    initial_margin = float(min_clearance(planner, problem['q_init'], problem['inputs']).cpu()[0])
    final_margin = float(min_clearance(planner, q_sol, inputs).cpu()[0])
    tcp_error = float(pos_err.cpu()[0])
    success = tcp_error < success_tcp_threshold_m and final_margin >= success_margin_threshold_m
    return {'tcp_error_m': tcp_error, 'rot_error_deg': float(rot_err.cpu()[0]), 'initial_margin_m': initial_margin, 'final_margin_m': final_margin, 'max_step_rad': float(max_step.cpu()), 'time_s': solve_time, 'success': int(success), 'success_tcp_threshold_m': float(success_tcp_threshold_m), 'success_margin_threshold_m': float(success_margin_threshold_m)}

def solve(planner, problem, verbose=False, *, success_tcp_threshold_m=DEFAULT_SUCCESS_TCP_THRESHOLD_M, success_margin_threshold_m=DEFAULT_SUCCESS_MARGIN_THRESHOLD_M):
    start = time.perf_counter()
    with torch.no_grad():
        q_sol, inputs, _ = planner.solve(problem['inputs'], damping=1.5, verbose=verbose)
        qdot = planner.qdot_trajectory_from_inputs(inputs)
    dt = time.perf_counter() - start
    return (q_sol, qdot, inputs, get_metrics(planner, problem, q_sol, inputs, dt, success_tcp_threshold_m=success_tcp_threshold_m, success_margin_threshold_m=success_margin_threshold_m))

def print_result(name, problem, q_sol, metrics):
    print(f'\nWynik: {name}')
    print(f"błąd TCP: {metrics['tcp_error_m']:.6f} m")
    print(f"margines końcowy: {metrics['final_margin_m']:+.6f} m")
    print(f"success: {bool(metrics.get('success', 0))}")
    print(f"czas: {metrics['time_s']:.3f} s")

def as_np(x):
    return x.detach().cpu().numpy()

def save_trajectory(path, problem, q_sol, qdot, extra_arrays=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'q_start': as_np(problem['q_start']), 'q_target': as_np(problem['q_target']), 'q_init': as_np(problem['q_init']), 'q_solution': as_np(q_sol), 'qdot_solution': as_np(qdot), 'tcp_goal_pos': as_np(problem['tcp_goal_pos']), 'tcp_goal_rot': as_np(problem['tcp_goal_rot']), 'tcp_link_name': np.asarray([TCP_LINK_NAME]), 'tool_transform': as_np(problem['tool_transform']), 'obstacle_centers': np.asarray([problem['obstacle_centers']], dtype=float), 'obstacle_radii': np.asarray([problem['obstacle_radii']], dtype=float), 'safety_margin': np.asarray([[SAFETY_MARGIN]], dtype=float), 'link_collision_radius': np.asarray([[LINK_COLLISION_RADIUS]], dtype=float), 'link_control_points': as_np(problem['link_points']), 'collision_link_names': np.asarray(COLLISION_LINK_NAMES), 'scenario_name': np.asarray([problem['scenario_name']]), 'goal_source': np.asarray([problem.get('goal_source', 'scenario')]), 'goal_rotation_active': np.asarray([[bool(problem.get('goal_rotation_active', True))]]), 'manual_target_ik_pos_error_m': np.asarray([[float(problem.get('manual_target_ik', {}).get('pos_error_m', np.nan)) if problem.get('manual_target_ik') is not None else np.nan]]), 'manual_target_ik_rot_error_deg': np.asarray([[float(problem.get('manual_target_ik', {}).get('rot_error_deg', np.nan)) if problem.get('manual_target_ik') is not None and problem.get('manual_target_ik', {}).get('rot_error_deg') is not None else np.nan]])}
    if problem.get('segment_name') is not None:
        payload['segment_name'] = np.asarray([problem['segment_name']])
    if extra_arrays:
        payload.update(extra_arrays)
    np.savez(path, **payload)
    pass

def make_plots(planner, problem, q_sol, inputs, prefix, label):
    plots = [(plot_tcp_trajectory_3d, (planner, problem['q_init'], q_sol, problem['tcp_goal_pos'], inputs), {'save_path': data_file(f'tcp_trajectory_{prefix}.png'), 'method_label': label}), (plot_tcp_error_over_time, (planner, problem['q_init'], q_sol, problem['tcp_goal_pos'], inputs), {'save_path': data_file(f'tcp_error_{prefix}.png'), 'method_label': label}), (plot_clearance_over_time, (planner, problem['q_init'], q_sol, inputs), {'save_path': data_file(f'clearance_{prefix}.png'), 'method_label': label}), (plot_clearance_per_link_over_time, (planner, problem['q_init'], q_sol, inputs), {'save_path': data_file(f'clearance_per_link_{prefix}.png'), 'method_label': label}), (plot_joint_trajectories, (problem['q_init'], q_sol), {'save_path': data_file(f'joint_trajectories_{prefix}.png'), 'method_label': label}), (plot_robot_start_goal_points, (planner, problem['q_init'], q_sol, inputs), {'save_path': data_file(f'link_control_point_snapshots_{prefix}.png')})]
    for fun, args, kwargs in plots:
        try:
            fun(*args, **kwargs)
        except TypeError:
            kwargs.pop('method_label', None)
            try:
                fun(*args, **kwargs)
            except Exception as e:
                pass
        except Exception as e:
            pass

def safe_bar_plot(names, values, ylabel, title, save_path):
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, values)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
        pass
    except Exception as e:
        pass

def plot_warmstart_comparison(results, save_prefix='comparison'):
    if len(results) < 2:
        return
    names = [r['label'] for r in results]
    safe_bar_plot(names=names, values=[r['metrics']['time_s'] for r in results], ylabel='Czas [s]', title='Porównanie czasu planowania', save_path=data_file(f'{save_prefix}_time.png'))
    safe_bar_plot(names=names, values=[r['metrics']['tcp_error_m'] for r in results], ylabel='Błąd TCP [m]', title='Porównanie błędu TCP', save_path=data_file(f'{save_prefix}_tcp_error.png'))
    safe_bar_plot(names=names, values=[r['metrics']['final_margin_m'] for r in results], ylabel='Margines końcowy [m]', title='Porównanie marginesu bezpieczeństwa', save_path=data_file(f'{save_prefix}_clearance.png'))
    try:
        import csv
        csv_path = data_file(f'{save_prefix}_metrics.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        keys = ['method', 'time_s', 'tcp_error_m', 'rot_error_deg', 'initial_margin_m', 'final_margin_m', 'max_step_rad', 'success', 'success_tcp_threshold_m', 'success_margin_threshold_m']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                row = {'method': r['label']}
                row.update(r['metrics'])
                writer.writerow({k: row.get(k, '') for k in keys})
        pass
    except Exception as e:
        pass

def save_metrics_csv(results, path='run_metrics.csv'):
    if not results:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    keys = ['method', 'time_s', 'tcp_error_m', 'rot_error_deg', 'initial_margin_m', 'final_margin_m', 'max_step_rad', 'success', 'success_tcp_threshold_m', 'success_margin_threshold_m']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            row = {'method': r['label']}
            row.update(r['metrics'])
            writer.writerow({k: row.get(k, '') for k in keys})
    pass

def print_neutral_comparison(results):
    if len(results) < 2:
        return
    pass
    pass
    for r in results:
        m = r['metrics']
        pass

def run_case(name, planner, problem, verbose=False, *, success_tcp_threshold_m=DEFAULT_SUCCESS_TCP_THRESHOLD_M, success_margin_threshold_m=DEFAULT_SUCCESS_MARGIN_THRESHOLD_M):
    label = 'MLP warm-start' if name == 'mlp' else 'linear warm-start'
    qdot_start = planner.qdot_trajectory_from_inputs(problem['inputs'])
    save_trajectory(root_file(f'trajectory_{name}_before_theseus.npz'), problem, problem['q_init'], qdot_start)
    q_sol, qdot, inputs, metrics = solve(planner, problem, verbose, success_tcp_threshold_m=success_tcp_threshold_m, success_margin_threshold_m=success_margin_threshold_m)
    print_result(label, problem, q_sol, metrics)
    save_trajectory(root_file(f'trajectory_{name}_after_theseus.npz'), problem, q_sol, qdot)
    save_trajectory(root_file('last_solution_trajectory.npz'), problem, q_sol, qdot)
    make_plots(planner, problem, q_sol, inputs, name, label)
    return {'name': name, 'label': label, 'q_sol': q_sol, 'qdot': qdot, 'inputs': inputs, 'metrics': metrics, 'problem': problem}

def _coerce_xyz_list(value, source_name: str):
    try:
        xyz = [float(value[0]), float(value[1]), float(value[2])]
    except Exception as exc:
        raise ValueError(f'Niepoprawny punkt w {source_name}: {value!r}. Oczekiwano trzech liczb X Y Z.') from exc
    return xyz

def parse_waypoints_file(path: str | Path):
    """Read waypoints from JSON, CSV, or whitespace-separated TXT.

    Accepted JSON formats:
      [[x, y, z], [x, y, z], ...]
      {"waypoints": [[x, y, z], ...]}

    Text/CSV format: one waypoint per line, either "x y z" or "x,y,z".
    Empty lines and lines starting with # are ignored.
    """
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if obj is not None:
        if isinstance(obj, dict):
            obj = obj.get('waypoints', obj.get('points', obj.get('targets')))
        if not isinstance(obj, (list, tuple)):
            raise ValueError(f'Plik {path} nie zawiera listy waypointów.')
        return [_coerce_xyz_list(item, str(path)) for item in obj]
    points = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if not raw or raw.startswith('#'):
            continue
        parts = raw.replace(',', ' ').split()
        if len(parts) < 3:
            raise ValueError(f'Niepoprawna linia {line_no} w {path}: {line!r}. Oczekiwano X Y Z.')
        points.append(_coerce_xyz_list(parts[:3], f'{path}:{line_no}'))
    return points

def resolve_waypoints(args):
    """Return an ordered list of manual TCP waypoint positions.

    --target-pos can be used as the first point when --waypoint-pos or
    --waypoints-file is also present. With only --target-pos, the script keeps
    the legacy single-goal behavior.
    """
    waypoints = []
    explicit_waypoint_mode = has_waypoint_targets(args)
    if explicit_waypoint_mode and args.target_pos is not None:
        waypoints.append(_coerce_xyz_list(args.target_pos, '--target-pos'))
    for item in getattr(args, 'waypoint_pos', None) or []:
        waypoints.append(_coerce_xyz_list(item, '--waypoint-pos'))
    if getattr(args, 'waypoints_file', None):
        waypoints.extend(parse_waypoints_file(args.waypoints_file))
    return waypoints

def concatenate_trajectories(tensors):
    if not tensors:
        raise ValueError('Brak segmentów trajektorii do połączenia.')
    pieces = [tensors[0]]
    pieces.extend((t[:, 1:, :] for t in tensors[1:]))
    return torch.cat(pieces, dim=1)

def make_sequence_problem(segment_results, q_init_full, q_solution_full):
    first = segment_results[0]['problem']
    last = segment_results[-1]['problem']
    problem = dict(last)
    problem['q_start'] = first['q_start']
    problem['q_target'] = last['q_target']
    problem['q_init'] = q_init_full
    problem['tcp_goal_pos'] = last['tcp_goal_pos']
    problem['tcp_goal_rot'] = last['tcp_goal_rot']
    problem['goal_source'] = 'manual_tcp_waypoint_sequence'
    problem['segment_name'] = 'waypoint_sequence'
    problem['inputs'] = last['inputs']
    return problem

def sequence_extra_arrays(segment_results, q_solution_full):
    lengths = [int(r['q_sol'].shape[1]) for r in segment_results]
    starts = []
    ends = []
    cursor = 0
    for i, length in enumerate(lengths):
        if i == 0:
            starts.append(0)
            ends.append(length - 1)
            cursor = length - 1
        else:
            starts.append(cursor)
            cursor += length - 1
            ends.append(cursor)
    return {'waypoint_tcp_goal_pos': np.concatenate([as_np(r['problem']['tcp_goal_pos']) for r in segment_results], axis=0), 'waypoint_tcp_goal_rot': np.concatenate([as_np(r['problem']['tcp_goal_rot']) for r in segment_results], axis=0), 'waypoint_q_target': np.concatenate([as_np(r['problem']['q_target']) for r in segment_results], axis=0), 'waypoint_segment_lengths_raw': np.asarray(lengths, dtype=np.int64), 'waypoint_segment_start_frame': np.asarray(starts, dtype=np.int64), 'waypoint_segment_end_frame': np.asarray(ends, dtype=np.int64), 'waypoint_tcp_error_m': np.asarray([r['metrics']['tcp_error_m'] for r in segment_results], dtype=float), 'waypoint_final_margin_m': np.asarray([r['metrics']['final_margin_m'] for r in segment_results], dtype=float), 'waypoint_success': np.asarray([r['metrics'].get('success', 0) for r in segment_results], dtype=np.int64), 'waypoint_count': np.asarray([[len(segment_results)]], dtype=np.int64), 'waypoint_total_frames': np.asarray([[int(q_solution_full.shape[1])]], dtype=np.int64), 'waypoint_labels': np.asarray([r['problem'].get('segment_name', f'waypoint_{i + 1:02d}') for i, r in enumerate(segment_results)])}

def run_waypoint_sequence_for_mode(mode, planner, scenario, args, waypoints, dtype, device, verbose=False, *, success_tcp_threshold_m=DEFAULT_SUCCESS_TCP_THRESHOLD_M, success_margin_threshold_m=DEFAULT_SUCCESS_MARGIN_THRESHOLD_M):
    label = 'MLP warm-start waypoint sequence' if mode == 'mlp' else 'linear warm-start waypoint sequence'
    pass
    segment_results = []
    current_q_start = None
    for idx, waypoint in enumerate(waypoints, start=1):
        segment_name = f'waypoint_{idx:02d}'
        pass
        problem = make_problem(planner, scenario, args, dtype, device, target_pos_override=waypoint, q_start_override=current_q_start, segment_name=segment_name)
        if problem.get('manual_target_ik') is not None:
            ik = problem['manual_target_ik']
            pass
            if ik.get('rot_error_deg') is not None:
                pass
        if mode == 'mlp':
            problem = load_mlp_warmstart(args, planner, problem, dtype, device)
            problem['segment_name'] = segment_name
        qdot_start = planner.qdot_trajectory_from_inputs(problem['inputs'])
        save_trajectory(root_file(f'trajectory_{mode}_{segment_name}_before_theseus.npz'), problem, problem['q_init'], qdot_start)
        q_sol, qdot, inputs, metrics = solve(planner, problem, verbose, success_tcp_threshold_m=success_tcp_threshold_m, success_margin_threshold_m=success_margin_threshold_m)
        print_result(f'{label} / {segment_name}', problem, q_sol, metrics)
        save_trajectory(root_file(f'trajectory_{mode}_{segment_name}_after_theseus.npz'), problem, q_sol, qdot)
        segment_results.append({'name': f'{mode}_{segment_name}', 'label': f'{label} / {segment_name}', 'q_sol': q_sol, 'qdot': qdot, 'inputs': inputs, 'metrics': metrics, 'problem': problem})
        current_q_start = q_sol[:, -1, :].detach().clone()
    q_init_full = concatenate_trajectories([r['problem']['q_init'] for r in segment_results])
    q_solution_full = concatenate_trajectories([r['q_sol'] for r in segment_results])
    qdot_full = planner.finite_difference_velocities(q_solution_full, planner.error_scales_gp_dt())
    sequence_problem = make_sequence_problem(segment_results, q_init_full, q_solution_full)
    total_time = sum((float(r['metrics']['time_s']) for r in segment_results))
    sequence_metrics = get_metrics(planner, sequence_problem, q_solution_full, sequence_problem['inputs'], total_time, success_tcp_threshold_m=success_tcp_threshold_m, success_margin_threshold_m=success_margin_threshold_m)
    sequence_metrics['success'] = int(all((int(r['metrics'].get('success', 0)) for r in segment_results)))
    sequence_metrics['num_waypoints'] = len(segment_results)
    extras = sequence_extra_arrays(segment_results, q_solution_full)
    combined_path = f'trajectory_{mode}_waypoints_after_theseus.npz'
    save_trajectory(root_file(combined_path), sequence_problem, q_solution_full, qdot_full, extras)
    save_trajectory(root_file('last_solution_trajectory.npz'), sequence_problem, q_solution_full, qdot_full, extras)
    pass
    pass
    pass
    pass
    pass
    return {'name': f'{mode}_waypoints', 'label': label, 'q_sol': q_solution_full, 'qdot': qdot_full, 'inputs': sequence_problem['inputs'], 'metrics': sequence_metrics, 'problem': sequence_problem, 'segments': segment_results}

def parse_args():
    parser = argparse.ArgumentParser(description='Prosty planner Panda 7DoF')
    parser.add_argument('--warmstart-mode', choices=('linear', 'mlp', 'both'), default='both')
    parser.add_argument('--mlp-checkpoint', default='dgpmp2_warmstart_max4.pt')
    parser.add_argument('--scenario', default=DEFAULT_SCENARIO_NAME, choices=sorted(SCENARIOS.keys()))
    parser.add_argument('--target-pos', nargs=3, type=float, metavar=('X', 'Y', 'Z'), default=None)
    parser.add_argument('--target-rpy', nargs=3, type=float, metavar=('ROLL', 'PITCH', 'YAW'), default=None)
    parser.add_argument('--waypoint-pos', nargs=3, type=float, action='append', metavar=('X', 'Y', 'Z'), default=None)
    parser.add_argument('--waypoints-file', default=None)
    parser.add_argument('--enforce-target-rotation', action='store_true', help='Przy --target-pos bez --target-rpy zachowaj i wymuszaj orientację TCP ze scenariusza. Domyślnie ręczny punkt jest position-only.')
    parser.add_argument('--max-iterations', type=int, default=MAX_ITERATIONS)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--success-tcp-threshold-m', type=float, default=DEFAULT_SUCCESS_TCP_THRESHOLD_M)
    parser.add_argument('--success-margin-threshold-m', type=float, default=DEFAULT_SUCCESS_MARGIN_THRESHOLD_M)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

def choose_device(name):
    if name == 'cpu':
        return torch.device('cpu')
    if name in ('cuda', 'auto') and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if name == 'auto' and major >= 12:
                pass
                pass
                return torch.device('cpu')
        except Exception:
            pass
        return torch.device('cuda')
    return torch.device('cpu')

def main():
    args = parse_args()
    torch.set_printoptions(precision=5, sci_mode=False)
    dtype = torch.double
    device = choose_device(args.device)
    scenario = get_scenario(args.scenario)
    waypoints = resolve_waypoints(args)
    if has_waypoint_targets(args) and (not waypoints):
        raise ValueError('Włączono tryb waypointów, ale nie podano żadnego poprawnego punktu XYZ.')
    pass
    pass
    pass
    pass
    pass
    pass
    if waypoints:
        pass
        for i, wp in enumerate(waypoints, start=1):
            pass
        if ignore_manual_goal_rotation(args, waypoints[0]):
            pass
        elif args.target_rpy is not None:
            pass
        else:
            pass
        planner = make_planner(args, scenario, dtype, device)
        modes = ['linear'] if args.warmstart_mode == 'linear' else ['mlp'] if args.warmstart_mode == 'mlp' else ['mlp', 'linear']
        results = []
        run_kwargs = {'success_tcp_threshold_m': args.success_tcp_threshold_m, 'success_margin_threshold_m': args.success_margin_threshold_m}
        for mode in modes:
            results.append(run_waypoint_sequence_for_mode(mode, planner, scenario, args, waypoints, dtype, device, args.verbose, **run_kwargs))
        save_metrics_csv(results, data_file('run_metrics.csv'))
        print_neutral_comparison(results)
        return
    if args.target_pos is not None:
        pass
        if ignore_manual_goal_rotation(args, args.target_pos):
            pass
        elif args.target_rpy is not None:
            pass
        else:
            pass
    planner = make_planner(args, scenario, dtype, device)
    base_problem = make_problem(planner, scenario, args, dtype, device)
    if base_problem.get('manual_target_ik') is not None:
        ik = base_problem['manual_target_ik']
        pass
        if ik.get('rot_error_deg') is not None:
            pass
    linear_problem = copy_problem(planner, base_problem, base_problem['q_init'].clone())
    mlp_problem = None
    if args.warmstart_mode in ('mlp', 'both'):
        mlp_problem = load_mlp_warmstart(args, planner, base_problem, dtype, device)
    results = []
    run_kwargs = {'success_tcp_threshold_m': args.success_tcp_threshold_m, 'success_margin_threshold_m': args.success_margin_threshold_m}
    if args.warmstart_mode in ('mlp', 'both'):
        results.append(run_case('mlp', planner, mlp_problem, args.verbose, **run_kwargs))
    if args.warmstart_mode in ('linear', 'both'):
        results.append(run_case('linear', planner, linear_problem, args.verbose, **run_kwargs))
    save_metrics_csv(results, data_file('run_metrics.csv'))
    print_neutral_comparison(results)
    plot_warmstart_comparison(results)
if __name__ == '__main__':
    main()
