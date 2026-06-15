"""Theseus trajectory optimizer for the cleaned Franka Panda baseline."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import theseus as th
except ImportError as exc:  # pragma: no cover
    raise ImportError("This project must run in an environment where `import theseus as th` works.") from exc

from .config import ErrorScales, PlannerWeights
from .costs import (
    gpmp2_constant_velocity_prior_error,
    joint_limit_error,
    joint_velocity_boundary_error,
    link_sphere_collision_error,
    start_joint_error,
    tcp_position_error,
    tcp_orientation_error,
)
from .kinematics import default_tool_transform, fk_urdf_tcp, link_control_points_world
from .robot_model import NUM_COLLISION_LINKS, RobotURDFKinematics, default_link_control_points
from .tensor_utils import TensorDict, tensor_from_value


class Theseus7DofPlannerMVP:
    """Trajectory optimizer for q_0 ... q_N of the 7-DOF Panda arm."""

    def __init__(
        self,
        num_steps: int = 32,
        weights: Optional[PlannerWeights] = None,
        error_scales: Optional[ErrorScales] = None,
        max_iterations: int = 100,
        step_size: float = 0.20,
        dtype: torch.dtype = torch.double,
        device: torch.device | str = "cpu",
        link_control_points_per_link: int = 25,
        num_obstacles: int = 2,
    ) -> None:
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        if torch.get_default_dtype() != dtype:
            torch.set_default_dtype(dtype)

        self.num_steps = num_steps
        self.weights = weights or PlannerWeights()
        self.error_scales = error_scales or ErrorScales()
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.link_control_points_per_link = link_control_points_per_link
        self.num_obstacles = num_obstacles
        self.num_collision_links = NUM_COLLISION_LINKS

        self.objective = th.Objective()
        self.q_vars = [th.Vector(7, name=f"q_{i}", dtype=dtype) for i in range(num_steps + 1)]
        self.qdot_vars = [th.Vector(7, name=f"qdot_{i}", dtype=dtype) for i in range(num_steps + 1)]
        self._make_aux_vars()
        self._build_objective()

        self.optimizer = th.LevenbergMarquardt(
            self.objective,
            th.CholeskyDenseSolver,
            max_iterations=max_iterations,
            step_size=step_size,
        )
        self.layer = th.TheseusLayer(self.optimizer)
        self.layer.to(device=self.device, dtype=self.dtype)

    def _make_aux_vars(self) -> None:
        d, dev = self.dtype, self.device
        robot = RobotURDFKinematics.franka_panda(dtype=d, device=dev)

        self.q_start = th.Variable(torch.zeros(1, 7, dtype=d, device=dev), name="q_start")
        self.qdot_start = th.Variable(torch.zeros(1, 7, dtype=d, device=dev), name="qdot_start")
        self.qdot_goal = th.Variable(torch.zeros(1, 7, dtype=d, device=dev), name="qdot_goal")
        self.tcp_goal_pos = th.Variable(torch.zeros(1, 3, dtype=d, device=dev), name="tcp_goal_pos")
        self.tcp_goal_rot = th.Variable(torch.eye(3, dtype=d, device=dev).view(1, 3, 3), name="tcp_goal_rot")
        self.joint_min = th.Variable(robot.joint_min.view(1, 7), name="joint_min")
        self.joint_max = th.Variable(robot.joint_max.view(1, 7), name="joint_max")
        self.joint_origin_xyz = th.Variable(robot.joint_origin_xyz.view(1, 7, 3), name="joint_origin_xyz")
        self.joint_origin_rpy = th.Variable(robot.joint_origin_rpy.view(1, 7, 3), name="joint_origin_rpy")
        self.joint_axis = th.Variable(robot.joint_axis.view(1, 7, 3), name="joint_axis")
        self.tcp_fixed_transform = th.Variable(robot.tcp_fixed_transform.view(1, 4, 4), name="tcp_fixed_transform")
        self.tool_transform = th.Variable(default_tool_transform(1, dtype=d, device=dev), name="tool_transform")

        self.obstacle_centers = th.Variable(torch.zeros(1, self.num_obstacles, 3, dtype=d, device=dev), name="obstacle_centers")
        self.obstacle_radii = th.Variable(torch.full((1, self.num_obstacles), 0.04, dtype=d, device=dev), name="obstacle_radii")
        self.collision_safety_margin = th.Variable(tensor_from_value(0.03, dtype=d, device=dev), name="collision_safety_margin")
        self.link_collision_radius = th.Variable(tensor_from_value(0.055, dtype=d, device=dev), name="link_collision_radius")
        self.link_control_points = th.Variable(
            default_link_control_points(points_per_link=self.link_control_points_per_link, dtype=d, device=dev),
            name="link_control_points",
        )

        s = self.error_scales
        self.tcp_pos_scale = th.Variable(tensor_from_value(s.tcp_pos_m, dtype=d, device=dev), name="tcp_pos_scale")
        self.tcp_rot_scale = th.Variable(tensor_from_value(s.tcp_rot_rad, dtype=d, device=dev), name="tcp_rot_scale")
        self.joint_scale = th.Variable(tensor_from_value(s.joint_rad, dtype=d, device=dev), name="joint_scale")
        self.qdot_scale = th.Variable(tensor_from_value(s.qdot_rad_s, dtype=d, device=dev), name="qdot_scale")
        self.gp_dt = th.Variable(tensor_from_value(getattr(s, "gp_dt", 0.10), dtype=d, device=dev), name="gp_dt")
        self.gp_qc = th.Variable(tensor_from_value(s.gp_qc, dtype=d, device=dev), name="gp_qc")
        self.limit_scale = th.Variable(tensor_from_value(s.limit_rad, dtype=d, device=dev), name="limit_scale")
        self.limit_margin = th.Variable(tensor_from_value(s.limit_margin_rad, dtype=d, device=dev), name="limit_margin")
        self.collision_scale = th.Variable(tensor_from_value(s.collision_m, dtype=d, device=dev), name="collision_scale")

    def _scale_weight(self, value: float, name: str):
        scale_var = th.Variable(torch.tensor([[value]], dtype=self.dtype, device=self.device), name=name)
        return th.ScaleCostWeight(scale_var)

    def _build_objective(self) -> None:
        w = self.weights

        self.objective.add(
            th.AutoDiffCostFunction(
                [self.q_vars[0]],
                start_joint_error,
                7,
                aux_vars=[self.q_start, self.joint_scale],
                cost_weight=self._scale_weight(w.start, "w_start"),
                name="start_joint_cost",
            )
        )
        self.objective.add(
            th.AutoDiffCostFunction(
                [self.q_vars[-1]],
                tcp_position_error,
                3,
                aux_vars=[
                    self.tcp_goal_pos,
                    self.joint_origin_xyz,
                    self.joint_origin_rpy,
                    self.joint_axis,
                    self.tcp_fixed_transform,
                    self.tool_transform,
                    self.tcp_pos_scale,
                ],
                cost_weight=self._scale_weight(w.tcp_pos, "w_tcp_pos"),
                name="tcp_position_cost",
            )
        )
        self.objective.add(
            th.AutoDiffCostFunction(
                [self.q_vars[-1]],
                tcp_orientation_error,
                3,
                aux_vars=[
                    self.tcp_goal_rot,
                    self.joint_origin_xyz,
                    self.joint_origin_rpy,
                    self.joint_axis,
                    self.tcp_fixed_transform,
                    self.tool_transform,
                    self.tcp_rot_scale,
                ],
                cost_weight=self._scale_weight(w.tcp_rot, "w_tcp_rot"),
                name="tcp_orientation_cost",
            )
        )

        self.objective.add(
            th.AutoDiffCostFunction(
                [self.qdot_vars[0]],
                joint_velocity_boundary_error,
                7,
                aux_vars=[self.qdot_start, self.qdot_scale],
                cost_weight=self._scale_weight(w.start_velocity, "w_start_velocity"),
                name="start_velocity_cost",
            )
        )
        self.objective.add(
            th.AutoDiffCostFunction(
                [self.qdot_vars[-1]],
                joint_velocity_boundary_error,
                7,
                aux_vars=[self.qdot_goal, self.qdot_scale],
                cost_weight=self._scale_weight(w.goal_velocity, "w_goal_velocity"),
                name="goal_velocity_cost",
            )
        )

        for i in range(self.num_steps):
            self.objective.add(
                th.AutoDiffCostFunction(
                    [self.q_vars[i], self.qdot_vars[i], self.q_vars[i + 1], self.qdot_vars[i + 1]],
                    gpmp2_constant_velocity_prior_error,
                    14,
                    aux_vars=[self.gp_dt, self.gp_qc],
                    cost_weight=self._scale_weight(w.gp_prior, f"w_gp_prior_{i}"),
                    name=f"gpmp2_prior_cost_{i}",
                )
            )

        # Check all knots, including q_0. start_joint_cost pins q_0 to q_start,
        # but it does not validate that q_start itself is inside the robot limits.
        for i in range(self.num_steps + 1):
            self.objective.add(
                th.AutoDiffCostFunction(
                    [self.q_vars[i]],
                    joint_limit_error,
                    14,
                    aux_vars=[self.joint_min, self.joint_max, self.limit_margin, self.limit_scale],
                    cost_weight=self._scale_weight(w.joint_limits, f"w_joint_limits_{i}"),
                    name=f"joint_limit_cost_{i}",
                )
            )

        link_collision_dim = self.num_obstacles * self.num_collision_links * self.link_control_points_per_link
        # Check every knot, including q_0 and q_N. Endpoint collisions are possible
        # even when q_0 is fixed and q_N is only constrained by the TCP goal cost.
        for i in range(self.num_steps + 1):
            self.objective.add(
                th.AutoDiffCostFunction(
                    [self.q_vars[i]],
                    link_sphere_collision_error,
                    link_collision_dim,
                    aux_vars=[
                        self.obstacle_centers,
                        self.obstacle_radii,
                        self.collision_safety_margin,
                        self.link_collision_radius,
                        self.joint_origin_xyz,
                        self.joint_origin_rpy,
                        self.joint_axis,
                        self.tcp_fixed_transform,
                        self.link_control_points,
                        self.collision_scale,
                    ],
                    cost_weight=self._scale_weight(w.link_collision, f"w_link_collision_{i}"),
                    name=f"link_sphere_collision_cost_{i}",
                )
            )

    @staticmethod
    def linear_interpolation(q_start: torch.Tensor, q_end: torch.Tensor, num_steps: int) -> torch.Tensor:
        if q_start.shape != q_end.shape or q_start.ndim != 2 or q_start.shape[1] != 7:
            raise ValueError("q_start and q_end must both have shape [B, 7]")
        B = q_start.shape[0]
        alphas = torch.linspace(0.0, 1.0, num_steps + 1, dtype=q_start.dtype, device=q_start.device)
        return (1.0 - alphas.view(1, -1, 1)) * q_start[:, None, :] + alphas.view(1, -1, 1) * q_end[:, None, :].expand(B, -1, -1)

    def error_scales_gp_dt(self) -> float:
        return float(getattr(self.error_scales, "gp_dt", 0.10))

    @staticmethod
    def finite_difference_velocities(q_traj: torch.Tensor, dt: float) -> torch.Tensor:
        if q_traj.ndim != 3 or q_traj.shape[-1] != 7:
            raise ValueError("q_traj must have shape [B, T, 7]")
        qdot = torch.zeros_like(q_traj)
        if q_traj.shape[1] > 2:
            qdot[:, 1:-1, :] = (q_traj[:, 2:, :] - q_traj[:, :-2, :]) / (2.0 * dt)
        if q_traj.shape[1] > 1:
            qdot[:, 0, :] = 0.0
            qdot[:, -1, :] = 0.0
        return qdot

    def _format_obstacle_centers(self, centers: torch.Tensor, batch_size: int) -> torch.Tensor:
        if centers.ndim == 2:
            centers = centers.unsqueeze(0)
        if centers.ndim != 3 or centers.shape[-1] != 3:
            raise ValueError(f"obstacle_centers must have shape [O,3] or [B,O,3], got {tuple(centers.shape)}")
        if centers.shape[0] == 1 and batch_size > 1:
            centers = centers.expand(batch_size, -1, -1)
        if centers.shape[0] != batch_size:
            raise ValueError(f"Obstacle batch {centers.shape[0]} incompatible with batch_size={batch_size}")
        if centers.shape[1] != self.num_obstacles:
            raise ValueError(f"Planner was built for {self.num_obstacles} obstacles, got {centers.shape[1]}")
        return centers

    def _format_obstacle_radii(self, radii: torch.Tensor, batch_size: int) -> torch.Tensor:
        if radii.ndim == 0:
            radii = radii.reshape(1, 1)
        elif radii.ndim == 1:
            radii = radii.reshape(1, -1)
        elif radii.ndim == 3 and radii.shape[-1] == 1:
            radii = radii.squeeze(-1)
        if radii.ndim != 2:
            raise ValueError(f"obstacle_radii must have shape [O], [B,O] or [B,O,1], got {tuple(radii.shape)}")
        if radii.shape[0] == 1 and batch_size > 1:
            radii = radii.expand(batch_size, -1)
        if radii.shape[0] != batch_size:
            raise ValueError(f"Obstacle radii batch {radii.shape[0]} incompatible with batch_size={batch_size}")
        if radii.shape[1] == 1 and self.num_obstacles > 1:
            radii = radii.expand(batch_size, self.num_obstacles)
        if radii.shape[1] != self.num_obstacles:
            raise ValueError(f"Planner was built for {self.num_obstacles} obstacles, got {radii.shape[1]} radii")
        return radii

    def make_inputs(
        self,
        *,
        q_start: torch.Tensor,
        tcp_goal_pos: torch.Tensor,
        q_init: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        collision_safety_margin: torch.Tensor,
        link_collision_radius: torch.Tensor,
        tcp_goal_rot: Optional[torch.Tensor] = None,
        qdot_init: Optional[torch.Tensor] = None,
        qdot_start: Optional[torch.Tensor] = None,
        qdot_goal: Optional[torch.Tensor] = None,
        robot_model: Optional[RobotURDFKinematics] = None,
        tool_transform: Optional[torch.Tensor] = None,
        link_control_points: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        q_start = q_start.to(device=self.device, dtype=self.dtype)
        tcp_goal_pos = tcp_goal_pos.to(device=self.device, dtype=self.dtype)
        tcp_goal_rot = torch.eye(3, dtype=self.dtype, device=self.device).view(1, 3, 3) if tcp_goal_rot is None else tcp_goal_rot.to(device=self.device, dtype=self.dtype)
        q_init = q_init.to(device=self.device, dtype=self.dtype)
        if q_init.ndim != 3 or q_init.shape[1:] != (self.num_steps + 1, 7):
            raise ValueError(f"q_init must have shape [B, {self.num_steps + 1}, 7]")

        if qdot_init is None:
            qdot_init = self.finite_difference_velocities(q_init, dt=self.error_scales_gp_dt())
        else:
            qdot_init = qdot_init.to(device=self.device, dtype=self.dtype)
        if qdot_init.ndim != 3 or qdot_init.shape[1:] != (self.num_steps + 1, 7):
            raise ValueError(f"qdot_init must have shape [B, {self.num_steps + 1}, 7]")
        qdot_start = torch.zeros_like(q_start) if qdot_start is None else qdot_start.to(device=self.device, dtype=self.dtype)
        qdot_goal = torch.zeros_like(q_start) if qdot_goal is None else qdot_goal.to(device=self.device, dtype=self.dtype)

        robot_model = robot_model or RobotURDFKinematics.franka_panda(dtype=self.dtype, device=self.device)
        tool_transform = default_tool_transform(1, dtype=self.dtype, device=self.device) if tool_transform is None else tool_transform.to(device=self.device, dtype=self.dtype)
        link_control_points = (
            default_link_control_points(points_per_link=self.link_control_points_per_link, dtype=self.dtype, device=self.device)
            if link_control_points is None
            else link_control_points.to(device=self.device, dtype=self.dtype)
        )
        if link_control_points.ndim != 4 or link_control_points.shape[1] != self.num_collision_links or link_control_points.shape[2] != self.link_control_points_per_link:
            raise ValueError(
                f"link_control_points must have shape [1 or B, {self.num_collision_links}, "
                f"{self.link_control_points_per_link}, 3], got {tuple(link_control_points.shape)}"
            )

        inputs: TensorDict = {
            "q_start": q_start,
            "tcp_goal_pos": tcp_goal_pos,
            "tcp_goal_rot": tcp_goal_rot,
            "qdot_start": qdot_start,
            "qdot_goal": qdot_goal,
            "joint_min": robot_model.joint_min.to(device=self.device, dtype=self.dtype).view(1, 7),
            "joint_max": robot_model.joint_max.to(device=self.device, dtype=self.dtype).view(1, 7),
            "joint_origin_xyz": robot_model.joint_origin_xyz.to(device=self.device, dtype=self.dtype).view(1, 7, 3),
            "joint_origin_rpy": robot_model.joint_origin_rpy.to(device=self.device, dtype=self.dtype).view(1, 7, 3),
            "joint_axis": robot_model.joint_axis.to(device=self.device, dtype=self.dtype).view(1, 7, 3),
            "tcp_fixed_transform": robot_model.tcp_fixed_transform.to(device=self.device, dtype=self.dtype).view(1, 4, 4),
            "tool_transform": tool_transform,
            "tcp_pos_scale": tensor_from_value(self.error_scales.tcp_pos_m, dtype=self.dtype, device=self.device),
            "tcp_rot_scale": tensor_from_value(self.error_scales.tcp_rot_rad, dtype=self.dtype, device=self.device),
            "joint_scale": tensor_from_value(self.error_scales.joint_rad, dtype=self.dtype, device=self.device),
            "qdot_scale": tensor_from_value(self.error_scales.qdot_rad_s, dtype=self.dtype, device=self.device),
            "gp_dt": tensor_from_value(self.error_scales_gp_dt(), dtype=self.dtype, device=self.device),
            "gp_qc": tensor_from_value(self.error_scales.gp_qc, dtype=self.dtype, device=self.device),
            "limit_scale": tensor_from_value(self.error_scales.limit_rad, dtype=self.dtype, device=self.device),
            "limit_margin": tensor_from_value(self.error_scales.limit_margin_rad, dtype=self.dtype, device=self.device),
            "collision_scale": tensor_from_value(self.error_scales.collision_m, dtype=self.dtype, device=self.device),
            "obstacle_centers": self._format_obstacle_centers(obstacle_centers.to(device=self.device, dtype=self.dtype), q_start.shape[0]),
            "obstacle_radii": self._format_obstacle_radii(obstacle_radii.to(device=self.device, dtype=self.dtype), q_start.shape[0]),
            "collision_safety_margin": collision_safety_margin.to(device=self.device, dtype=self.dtype).reshape(-1, 1),
            "link_collision_radius": link_collision_radius.to(device=self.device, dtype=self.dtype).reshape(-1, 1),
            "link_control_points": link_control_points,
        }
        for i in range(self.num_steps + 1):
            inputs[f"q_{i}"] = q_init[:, i, :]
            inputs[f"qdot_{i}"] = qdot_init[:, i, :]
        return inputs

    def solve(
        self,
        inputs: TensorDict,
        *,
        damping: float = 1.5,
        verbose: bool = False,
        track_best_solution: bool = True,
        backward_mode: str | None = None,
    ) -> Tuple[torch.Tensor, TensorDict, object]:
        optimizer_kwargs = {
            "damping": damping,
            "verbose": verbose,
            "track_best_solution": track_best_solution,
        }
        if backward_mode is not None:
            optimizer_kwargs["backward_mode"] = backward_mode

        updated_optim_vars, info = self.layer.forward(
            inputs,
            optimizer_kwargs=optimizer_kwargs,
        )

        # TheseusLayer returns the optimized variables, but not necessarily all auxiliary
        # variables that were passed in the input dictionary. Plotting, saving and PyBullet validation still need robot constants,
        # obstacle parameters and scales, so keep them in the updated dict.
        full_updated_inputs = dict(inputs)
        full_updated_inputs.update(updated_optim_vars)

        q_traj = torch.stack([full_updated_inputs[f"q_{i}"] for i in range(self.num_steps + 1)], dim=1)
        return q_traj, full_updated_inputs, info

    def qdot_trajectory_from_inputs(self, inputs: TensorDict) -> torch.Tensor:
        return torch.stack([inputs[f"qdot_{i}"] for i in range(self.num_steps + 1)], dim=1)

    def tcp_pose(self, q: torch.Tensor, inputs: TensorDict) -> torch.Tensor:
        return fk_urdf_tcp(
            q.to(device=self.device, dtype=self.dtype),
            inputs["joint_origin_xyz"],
            inputs["joint_origin_rpy"],
            inputs["joint_axis"],
            inputs["tcp_fixed_transform"],
            inputs["tool_transform"],
        )

    def link_control_points_world(self, q: torch.Tensor, inputs: TensorDict) -> torch.Tensor:
        return link_control_points_world(
            q.to(device=self.device, dtype=self.dtype),
            inputs["joint_origin_xyz"],
            inputs["joint_origin_rpy"],
            inputs["joint_axis"],
            inputs["link_control_points"],
            inputs["tcp_fixed_transform"],
        )
