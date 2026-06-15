"""Training script for the Franka Panda dGPMP2 MLP warm-start.

The MLP predicts a bounded correction to a linear joint-space interpolation.
Theseus/GPMP2 still optimizes the final trajectory.  This version keeps the
hand-written benchmark scenes, but also builds randomized train/validation/test
splits so the reported numbers are no longer tied to one fixed q_start.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch

from run_planner import data_file, root_file, save_trajectory
from franka_dgpmp2.config import (
    LINK_COLLISION_RADIUS,
    Q_START,
    SAFETY_MARGIN,
    TCP_LINK_NAME,
    PlannerWeights,
    SCENARIOS,
)
from franka_dgpmp2.costs import _so3_log_vector
from franka_dgpmp2.kinematics import fk_urdf_tcp, tool_transform_for_tcp_link
from franka_dgpmp2.planner import Theseus7DofPlannerMVP
from franka_dgpmp2.robot_model import RobotURDFKinematics, default_link_control_points
from franka_dgpmp2.warmstart_mlp import WarmStartMLP


MAX_OBSTACLES = 4
FAR_AWAY = (10.0, 10.0, 10.0)
DEFAULT_SUCCESS_TCP_THRESHOLD_M = 0.005
DEFAULT_SUCCESS_MARGIN_THRESHOLD_M = 0.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VAL_TRAJECTORY = "last_dgpmp2_generalization_val_trajectory.npz"

# Workspace ranges used by the synthetic obstacle generator.  They are broad
# enough for the demo scenes, but intentionally conservative so random scenes
# remain plausible for a tabletop Panda setup.
OBSTACLE_CENTER_LOW = (0.05, -0.35, 0.55)
OBSTACLE_CENTER_HIGH = (0.55, 0.40, 0.95)
STRESS_CENTER_LOW = (0.10, -0.25, 0.60)
STRESS_CENTER_HIGH = (0.48, 0.32, 0.88)


def _tuple7(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != 7:
        raise ValueError(f"Expected 7 joint values, got {len(values)}")
    return tuple(float(v) for v in values)


def _tuple_centers(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    centers = []
    for c in values:
        if len(c) != 3:
            raise ValueError(f"Obstacle center must have 3 values, got {len(c)}")
        centers.append((float(c[0]), float(c[1]), float(c[2])))
    return tuple(centers)


def scenario(name: str) -> dict:
    s = SCENARIOS[name]
    return {
        "name": name,
        "q_start": _tuple7(s.q_start),
        "q_target": _tuple7(s.q_target),
        "centers": _tuple_centers(s.obstacle_centers),
        "radii": tuple(float(r) for r in s.obstacle_radii),
        "source": "fixed",
    }


def custom(
    name: str,
    q_target: Sequence[float],
    centers: Sequence[Sequence[float]],
    radii: Sequence[float],
    *,
    q_start: Sequence[float] = Q_START,
    source: str = "fixed",
) -> dict:
    return {
        "name": name,
        "q_start": _tuple7(q_start),
        "q_target": _tuple7(q_target),
        "centers": _tuple_centers(centers),
        "radii": tuple(float(r) for r in radii),
        "source": source,
    }


# Fixed scenes used during training.  s8_clutter remains a train case, so it is
# deliberately absent from BASE_VAL_CASES to avoid validation leakage.
BASE_TRAIN_CASES = [
    scenario("s1_easy_reach"),
    scenario("s6_high_reach"),
    scenario("s2_central_obstacle"),
    scenario("s3_narrow_passage"),
    scenario("s4_goal_near_obstacle"),
    scenario("s5_around_back"),
    scenario("s8_clutter"),
    custom(
        "synthetic_three_obstacles",
        SCENARIOS["s8_clutter"].q_target,
        SCENARIOS["s8_clutter"].obstacle_centers[:3],
        SCENARIOS["s8_clutter"].obstacle_radii[:3],
    ),
]

BASE_VAL_CASES = [
    scenario("s7_low_reach"),
    custom(
        "val_three_obstacles",
        (1.00, -1.25, 0.42, -2.20, 0.68, 2.40, 1.10),
        SCENARIOS["s8_clutter"].obstacle_centers[:3],
        SCENARIOS["s8_clutter"].obstacle_radii[:3],
    ),
]


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Wybrano CUDA, ale torch.cuda.is_available() == False")
    return torch.device(name)


def make_planner(args, dtype: torch.dtype, device: torch.device) -> Theseus7DofPlannerMVP:
    return Theseus7DofPlannerMVP(
        num_steps=args.train_steps,
        weights=PlannerWeights(),
        max_iterations=args.inner_iters,
        step_size=0.20,
        dtype=dtype,
        device=device,
        link_control_points_per_link=args.points_per_link,
        num_obstacles=MAX_OBSTACLES,
    )


def _joint_limits() -> tuple[list[float], list[float]]:
    robot = RobotURDFKinematics.franka_panda(dtype=torch.double, device="cpu")
    return robot.joint_min.tolist(), robot.joint_max.tolist()


def _sample_joint_vector(rng: random.Random, *, margin: float = 0.08) -> tuple[float, ...]:
    joint_min, joint_max = _joint_limits()
    values = []
    for lo, hi in zip(joint_min, joint_max):
        span = hi - lo
        lo2 = lo + margin * span
        hi2 = hi - margin * span
        values.append(rng.uniform(lo2, hi2))
    return tuple(values)


def _sample_obstacles(
    rng: random.Random,
    *,
    max_obstacles: int,
    center_low: Sequence[float],
    center_high: Sequence[float],
    radius_range: tuple[float, float],
    min_obstacles: int = 1,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, ...]]:
    n_obs = rng.randint(min_obstacles, max_obstacles)
    centers = []
    for _ in range(n_obs):
        centers.append(tuple(rng.uniform(lo, hi) for lo, hi in zip(center_low, center_high)))
    radii = tuple(rng.uniform(radius_range[0], radius_range[1]) for _ in range(n_obs))
    return tuple(centers), radii


def random_case(rng: random.Random, name: str, *, stress: bool = False) -> dict:
    if stress:
        centers, radii = _sample_obstacles(
            rng,
            max_obstacles=MAX_OBSTACLES,
            min_obstacles=max(2, MAX_OBSTACLES - 1),
            center_low=STRESS_CENTER_LOW,
            center_high=STRESS_CENTER_HIGH,
            radius_range=(0.045, 0.075),
        )
        joint_margin = 0.03
    else:
        centers, radii = _sample_obstacles(
            rng,
            max_obstacles=MAX_OBSTACLES,
            center_low=OBSTACLE_CENTER_LOW,
            center_high=OBSTACLE_CENTER_HIGH,
            radius_range=(0.030, 0.060),
        )
        joint_margin = 0.08

    return custom(
        name,
        q_target=_sample_joint_vector(rng, margin=joint_margin),
        q_start=_sample_joint_vector(rng, margin=joint_margin),
        centers=centers,
        radii=radii,
        source="stress_random" if stress else "random",
    )


def generate_random_cases(count: int, *, seed: int, prefix: str, stress: bool = False) -> list[dict]:
    rng = random.Random(seed)
    return [random_case(rng, f"{prefix}_{i:03d}", stress=stress) for i in range(max(0, count))]


def build_case_splits(args) -> dict[str, list[dict]]:
    train_cases = list(BASE_TRAIN_CASES)
    validation_cases = list(BASE_VAL_CASES)

    train_cases += generate_random_cases(args.random_train_cases, seed=args.seed + 11, prefix="random_train")
    validation_cases += generate_random_cases(args.random_val_cases, seed=args.seed + 22, prefix="random_validation")
    random_test_cases = generate_random_cases(args.random_test_cases, seed=args.seed + 33, prefix="random_test")
    stress_test_cases = generate_random_cases(args.stress_test_cases, seed=args.seed + 44, prefix="stress_test", stress=True)

    train_names = {c["name"] for c in train_cases}
    leaked = [c["name"] for c in validation_cases if c["name"] in train_names]
    if leaked:
        raise RuntimeError(f"Validation leakage: {leaked}")

    return {
        "train": train_cases,
        "validation": validation_cases,
        "random_test": random_test_cases,
        "stress_test": stress_test_cases,
    }


def pad_obstacles(centers, radii, dtype: torch.dtype, device: torch.device, *, for_planner: bool = False):
    centers = torch.tensor(centers, dtype=dtype, device=device).reshape(-1, 3)
    radii = torch.tensor(radii, dtype=dtype, device=device).reshape(-1)

    if len(centers) != len(radii):
        raise ValueError("Liczba środków i promieni przeszkód się nie zgadza")
    if len(centers) > MAX_OBSTACLES:
        raise ValueError(f"Za dużo przeszkód, max = {MAX_OBSTACLES}")

    if for_planner:
        out_centers = torch.tensor(FAR_AWAY, dtype=dtype, device=device).repeat(MAX_OBSTACLES, 1)
    else:
        out_centers = torch.zeros(MAX_OBSTACLES, 3, dtype=dtype, device=device)

    out_radii = torch.zeros(MAX_OBSTACLES, dtype=dtype, device=device)
    out_centers[: len(centers)] = centers
    out_radii[: len(radii)] = radii
    return out_centers, out_radii


def make_problem(planner, case: dict, dtype: torch.dtype, device: torch.device) -> dict:
    robot = RobotURDFKinematics.franka_panda(dtype=dtype, device=device)
    tool = tool_transform_for_tcp_link(TCP_LINK_NAME, 1, dtype=dtype, device=device)

    q_start = torch.tensor([case["q_start"]], dtype=dtype, device=device)
    q_target = torch.tensor([case["q_target"]], dtype=dtype, device=device)
    q_init = planner.linear_interpolation(q_start, q_target, planner.num_steps)

    T_goal = fk_urdf_tcp(
        q_target,
        robot.joint_origin_xyz.view(1, 7, 3),
        robot.joint_origin_rpy.view(1, 7, 3),
        robot.joint_axis.view(1, 7, 3),
        robot.tcp_fixed_transform.view(1, 4, 4),
        tool,
    )

    feat_c, feat_r = pad_obstacles(case["centers"], case["radii"], dtype, device, for_planner=False)
    plan_c, plan_r = pad_obstacles(case["centers"], case["radii"], dtype, device, for_planner=True)

    return {
        "name": case["name"],
        "source": case.get("source", "fixed"),
        "robot": robot,
        "tool_transform": tool,
        "q_start": q_start,
        "q_target": q_target,
        "q_init": q_init,
        "tcp_goal_pos": T_goal[:, :3, 3],
        "tcp_goal_rot": T_goal[:, :3, :3],
        "link_points": default_link_control_points(
            points_per_link=planner.link_control_points_per_link,
            dtype=dtype,
            device=device,
        ),
        "obstacle_centers": case["centers"],
        "obstacle_radii": case["radii"],
        "feature_centers": feat_c,
        "feature_radii": feat_r,
        "planner_centers": plan_c,
        "planner_radii": plan_r,
        "scenario_name": case["name"],
    }


def features(problem: dict) -> torch.Tensor:
    q_start = problem["q_start"].reshape(1, -1)
    q_target = problem["q_target"].reshape(1, -1)
    q_delta = q_target - q_start
    centers = problem["feature_centers"].reshape(1, -1)
    radii = problem["feature_radii"].reshape(1, -1)
    return torch.cat([q_start, q_target, q_delta, centers, radii], dim=1)


def finite_diff(q: torch.Tensor, dt: float) -> torch.Tensor:
    zeros = torch.zeros_like(q[:, :1, :])
    if q.shape[1] <= 2:
        return torch.cat([zeros, zeros], dim=1)[:, : q.shape[1], :]
    middle = (q[:, 2:, :] - q[:, :-2, :]) / (2.0 * dt)
    return torch.cat([zeros, middle, zeros], dim=1)


def make_inputs(planner, problem: dict, q_init: torch.Tensor, dt: float):
    return planner.make_inputs(
        q_start=problem["q_start"],
        q_init=q_init,
        qdot_init=finite_diff(q_init, dt),
        tcp_goal_pos=problem["tcp_goal_pos"],
        tcp_goal_rot=problem["tcp_goal_rot"],
        robot_model=problem["robot"],
        tool_transform=problem["tool_transform"],
        obstacle_centers=problem["planner_centers"].view(1, MAX_OBSTACLES, 3),
        obstacle_radii=problem["planner_radii"].view(1, MAX_OBSTACLES),
        collision_safety_margin=torch.tensor([[SAFETY_MARGIN]], dtype=q_init.dtype, device=q_init.device),
        link_collision_radius=torch.tensor([[LINK_COLLISION_RADIUS]], dtype=q_init.dtype, device=q_init.device),
        link_control_points=problem["link_points"],
    )


def solve(planner, problem: dict, q_init: torch.Tensor, dt: float, backward_mode=None, track_best: bool = False, verbose: bool = False):
    inputs = make_inputs(planner, problem, q_init, dt)
    q_solution, new_inputs, _ = planner.solve(
        inputs,
        damping=1.5,
        verbose=verbose,
        backward_mode=backward_mode,
        track_best_solution=track_best,
    )
    return q_solution, new_inputs


def link_margins(planner, q_traj: torch.Tensor, inputs) -> torch.Tensor:
    centers = inputs["obstacle_centers"]
    radii = inputs["obstacle_radii"]
    safety = inputs["collision_safety_margin"].reshape(-1, 1, 1, 1)
    link_r = inputs["link_collision_radius"].reshape(-1, 1, 1, 1)

    batch, steps = q_traj.shape[:2]
    points = planner.link_control_points_world(q_traj.reshape(-1, 7), inputs)
    points = points.reshape(batch, steps, points.shape[1], points.shape[2], 3)

    dist = torch.linalg.norm(points[:, :, None] - centers[:, None, :, None, None], dim=-1)
    required = radii[:, None, :, None, None] + safety + link_r
    return dist - required


def loss_and_metrics(planner, q_solution: torch.Tensor, inputs, problem: dict, delta: torch.Tensor | None = None):
    T = planner.tcp_pose(q_solution[:, -1, :], inputs)
    pos_err = T[:, :3, 3] - problem["tcp_goal_pos"]
    rot_err = T[:, :3, :3] - problem["tcp_goal_rot"]

    R_err = torch.matmul(problem["tcp_goal_rot"].transpose(1, 2), T[:, :3, :3])
    rot_vec = _so3_log_vector(R_err).detach()

    margins = link_margins(planner, q_solution, inputs)

    ## ZMIANA
    target_margin = 0.005  # 5 mm clearance target
    violation = torch.clamp(target_margin - margins, min=0.0)
    flat_violation = violation.reshape(violation.shape[0], -1)
    k = max(1, int(0.05 * flat_violation.shape[1]))
    collision = flat_violation.pow(2).topk(k, dim=1).values.mean()

    #collision = torch.clamp(-margins, min=0.0).pow(2).mean()

    ##
    smooth = (q_solution[:, 1:, :] - q_solution[:, :-1, :]).pow(2).mean()
    delta_reg = torch.zeros((), dtype=q_solution.dtype, device=q_solution.device) if delta is None else delta.pow(2).mean()

    ## ZMIANA
    loss = 1000.0 * pos_err.pow(2).mean() + 10.0 * rot_err.pow(2).mean() + 200.0 * collision + 0.05 * smooth + 0.01 * delta_reg
    #loss = 1000.0 * pos_err.pow(2).mean() + 10.0 * rot_err.pow(2).mean() + 50.0 * collision + 0.05 * smooth + 0.01 * delta_reg
    ##

    return loss, {
        "loss": float(loss.detach().cpu()),
        "tcp_error_m": float(torch.linalg.norm(pos_err, dim=1).mean().detach().cpu()),
        "rot_error_deg": float(torch.rad2deg(torch.linalg.norm(rot_vec, dim=1).mean()).detach().cpu()),
        "min_margin_m": float(margins.amin().detach().cpu()),
        "collision_penalty": float(collision.detach().cpu()),
        "step_smoothness": float(smooth.detach().cpu()),
    }


def add_success_metrics(metrics: dict, args) -> dict:
    enriched = dict(metrics)
    success = (
        enriched["tcp_error_m"] < args.success_tcp_threshold_m
        and enriched["min_margin_m"] >= args.success_margin_threshold_m
    )
    enriched["success"] = int(success)
    enriched["success_tcp_threshold_m"] = float(args.success_tcp_threshold_m)
    enriched["success_margin_threshold_m"] = float(args.success_margin_threshold_m)
    return enriched


def grad_norm(model: torch.nn.Module) -> tuple[float, bool]:
    total = 0.0
    ok = True
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            ok = False
            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        total += float(p.grad.detach().pow(2).sum().cpu())
    return math.sqrt(total), ok


def save_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def learned_init(model: WarmStartMLP, problem: dict, max_delta: float):
    q_init = problem["q_init"].detach().clone()
    delta = model(features(problem), max_delta=max_delta)
    q_init[:, 1:-1, :] += delta
    return q_init, delta


def evaluate(model, planner, cases: Iterable[dict], split: str, args, dtype: torch.dtype, device: torch.device, dt: float, backward_mode):
    rows = []
    last_val = None

    for i, case in enumerate(cases):
        problem = make_problem(planner, case, dtype, device)

        with torch.no_grad():
            q_learned, _ = learned_init(model, problem, args.max_delta)

        for method, q_init in [("linear", problem["q_init"].detach()), ("learned", q_learned)]:
            with torch.no_grad():
                q_sol, inputs = solve(planner, problem, q_init, dt, backward_mode, track_best=True)
                _, metrics = loss_and_metrics(planner, q_sol, inputs, problem)

            row = {
                "split": split,
                "case_id": i,
                "case_name": case["name"],
                "case_source": case.get("source", "fixed"),
                "num_obstacles": len(case["radii"]),
                "method": method,
                **add_success_metrics(metrics, args),
            }
            rows.append(row)

            if split == "validation" and i == 0 and method == "learned":
                qdot = planner.qdot_trajectory_from_inputs(inputs).detach()
                last_val = (problem, q_sol.detach(), qdot)

    return rows, last_val


def summarize_eval(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["method"])].append(row)

    summary = []
    for (split, method), items in sorted(groups.items()):
        n = len(items)
        success_count = sum(int(r["success"]) for r in items)
        summary.append(
            {
                "split": split,
                "method": method,
                "num_cases": n,
                "success_count": success_count,
                "success_rate": success_count / n if n else 0.0,
                "mean_tcp_error_m": sum(float(r["tcp_error_m"]) for r in items) / n if n else 0.0,
                "mean_min_margin_m": sum(float(r["min_margin_m"]) for r in items) / n if n else 0.0,
                "mean_loss": sum(float(r["loss"]) for r in items) / n if n else 0.0,
            }
        )
    return summary


def print_eval(rows: list[dict]) -> None:
    print("\n=== Evaluation cases ===")
    for r in rows:
        print(
            f"{r['split']:11s} {r['case_id']:03d} {r['method']:7s} | "
            f"success={r['success']} | loss={r['loss']:.5f} | "
            f"tcp={r['tcp_error_m']:.3e} m | rot={r['rot_error_deg']:.3e} deg | "
            f"margin={r['min_margin_m']:+.4f} m | source={r['case_source']}"
        )


def print_summary(rows: list[dict]) -> None:
    print("\n=== Evaluation summary ===")
    for r in rows:
        print(
            f"{r['split']:11s} {r['method']:7s} | "
            f"success_rate={100.0 * r['success_rate']:.1f}% "
            f"({r['success_count']}/{r['num_cases']}) | "
            f"mean_tcp={r['mean_tcp_error_m']:.3e} m | "
            f"mean_margin={r['mean_min_margin_m']:+.4f} m"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Trening MLP warm-start dla dGPMP2 na scenach fixed + random.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--inner-iters", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-delta", type=float, default=0.20)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--train-steps", type=int, default=12)
    p.add_argument("--points-per-link", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--backward-mode", choices=("none", "unroll", "implicit", "dlm", "truncated"), default="unroll")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--random-train-cases", type=int, default=16)
    p.add_argument("--random-val-cases", type=int, default=6)
    p.add_argument("--random-test-cases", type=int, default=20)
    p.add_argument("--stress-test-cases", type=int, default=10)
    p.add_argument("--success-tcp-threshold-m", type=float, default=DEFAULT_SUCCESS_TCP_THRESHOLD_M)
    p.add_argument("--success-margin-threshold-m", type=float, default=DEFAULT_SUCCESS_MARGIN_THRESHOLD_M)
    p.add_argument("--checkpoint", default="dgpmp2_warmstart_max4.pt")
    p.add_argument("--train-csv", default="dgpmp2_train_log.csv")
    p.add_argument("--eval-csv", default="dgpmp2_eval.csv")
    p.add_argument("--eval-summary-csv", default="dgpmp2_eval_summary.csv")
    p.add_argument("--log-every", type=int, default=1, help="Print every N training steps; use 0 to print only epoch summaries.")
    p.add_argument("--no-shuffle-train", action="store_true", help="Keep training cases in fixed order within each epoch.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = root_file(args.checkpoint)
    args.train_csv = data_file(args.train_csv)
    args.eval_csv = data_file(args.eval_csv)
    args.eval_summary_csv = data_file(args.eval_summary_csv)
    val_trajectory = root_file(DEFAULT_VAL_TRAJECTORY)

    dtype = torch.double
    device = choose_device(args.device)
    backward_mode = None if args.backward_mode == "none" else args.backward_mode

    torch.manual_seed(args.seed)
    torch.set_printoptions(precision=5, sci_mode=False)

    case_splits = build_case_splits(args)
    planner = make_planner(args, dtype, device)
    dt = float(planner.error_scales_gp_dt())

    example = make_problem(planner, case_splits["train"][0], dtype, device)
    input_dim = features(example).shape[1]
    internal_knots = args.train_steps - 1

    model = WarmStartMLP(input_dim, internal_knots, hidden_dim=args.hidden_dim).to(dtype=dtype, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("=== dGPMP2 MLP warm-start ===")
    print(f"device:       {device}")
    print(f"checkpointy/trajektorie: {SCRIPT_DIR}")
    print(f"CSV:                   {SCRIPT_DIR / 'data'}")
    print(f"seed:         {args.seed}")
    total_train_steps = args.epochs * len(case_splits["train"])
    print(f"epochs:       {args.epochs} full passes over train cases")
    print(f"train steps:  {total_train_steps}")
    print(f"inner iters:  {args.inner_iters}")
    print(f"train cases:  {len(case_splits['train'])}")
    print(f"validation:   {len(case_splits['validation'])}")
    print(f"random test:  {len(case_splits['random_test'])}")
    print(f"stress test:  {len(case_splits['stress_test'])}")
    print(f"success:      tcp < {args.success_tcp_threshold_m} m and margin >= {args.success_margin_threshold_m} m\n")

    history = []
    train_cases = case_splits["train"]
    global_step = 0

    for epoch in range(args.epochs):
        case_order = list(range(len(train_cases)))
        if not args.no_shuffle_train:
            random.Random(args.seed + 1000 + epoch).shuffle(case_order)

        epoch_rows = []
        for epoch_case_index, case_id in enumerate(case_order):
            case = train_cases[case_id]
            problem = make_problem(planner, case, dtype, device)

            optimizer.zero_grad(set_to_none=True)
            q_init, delta = learned_init(model, problem, args.max_delta)
            q_sol, inputs = solve(planner, problem, q_init, dt, backward_mode, verbose=args.verbose)
            loss, metrics = loss_and_metrics(planner, q_sol, inputs, problem, delta)
            metrics = add_success_metrics(metrics, args)

            if not loss.requires_grad:
                raise RuntimeError(
                    "Loss nie jest połączony z MLP. Użyj różniczkowalnego --backward-mode, np. unroll albo implicit."
                )

            loss.backward()
            gnorm, ok = grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if ok:
                optimizer.step()

            row = {
                "step": global_step,
                "epoch": epoch,
                "epoch_case_index": epoch_case_index,
                "case_id": case_id,
                "case_name": case["name"],
                "case_source": case.get("source", "fixed"),
                "num_obstacles": len(case["radii"]),
                **metrics,
                "grad_norm": gnorm,
                "grad_ok": ok,
            }
            history.append(row)
            epoch_rows.append(row)

            should_log_step = args.log_every > 0 and (
                global_step % args.log_every == 0 or epoch_case_index == len(case_order) - 1
            )
            if should_log_step:
                print(
                    f"step {global_step:05d} epoch {epoch:03d} "
                    f"case={case_id:03d}/{epoch_case_index:03d} {case['name']} | "
                    f"success={metrics['success']} | loss={metrics['loss']:.5f} | "
                    f"tcp={metrics['tcp_error_m']:.3e} m | rot={metrics['rot_error_deg']:.3e} deg | "
                    f"margin={metrics['min_margin_m']:+.4f} m | grad={gnorm:.3e} | ok={ok}"
                )

            global_step += 1

        n_epoch = len(epoch_rows)
        mean_loss = sum(float(r["loss"]) for r in epoch_rows) / n_epoch
        mean_tcp = sum(float(r["tcp_error_m"]) for r in epoch_rows) / n_epoch
        mean_margin = sum(float(r["min_margin_m"]) for r in epoch_rows) / n_epoch
        success_rate = sum(int(r["success"]) for r in epoch_rows) / n_epoch
        print(
            f"epoch {epoch:03d} summary | "
            f"success_rate={100.0 * success_rate:.1f}% ({sum(int(r['success']) for r in epoch_rows)}/{n_epoch}) | "
            f"mean_loss={mean_loss:.5f} | mean_tcp={mean_tcp:.3e} m | "
            f"mean_margin={mean_margin:+.4f} m"
        )

    save_csv(history, args.train_csv)
    print(f"\nSaved train log: {args.train_csv}")

    eval_rows = []
    last_val = None
    for split in ("train", "validation", "random_test", "stress_test"):
        rows, maybe_last = evaluate(model, planner, case_splits[split], split, args, dtype, device, dt, backward_mode)
        eval_rows.extend(rows)
        if maybe_last is not None:
            last_val = maybe_last

    summary_rows = summarize_eval(eval_rows)
    save_csv(eval_rows, args.eval_csv)
    save_csv(summary_rows, args.eval_summary_csv)
    print(f"Saved eval table: {args.eval_csv}")
    print(f"Saved eval summary: {args.eval_summary_csv}")
    print_eval(eval_rows)
    print_summary(summary_rows)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "dof": 7,
        "train_steps": args.train_steps,
        "num_internal_knots": internal_knots,
        "max_delta": args.max_delta,
        "max_obstacles": MAX_OBSTACLES,
        "feature_description": "q_start, q_target, q_delta, obstacle_centers, obstacle_radii",
        "seed": args.seed,
        "success_tcp_threshold_m": args.success_tcp_threshold_m,
        "success_margin_threshold_m": args.success_margin_threshold_m,
        "num_train_cases": len(train_cases),
        "total_train_steps": global_step,
        "epochs_are_full_passes": True,
        "shuffle_train_cases": not args.no_shuffle_train,
        "case_splits": case_splits,
    }
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    print(f"Saved checkpoint: {args.checkpoint}")

    if last_val is not None:
        problem, q_sol, qdot = last_val
        save_trajectory(val_trajectory, problem, q_sol, qdot)
        print(f"Saved validation trajectory: {val_trajectory}")


if __name__ == "__main__":
    main()
