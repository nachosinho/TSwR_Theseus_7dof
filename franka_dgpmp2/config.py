"""Configuration and benchmark scenes for the dGPMP2 project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PlannerWeights:
    """Weights used by Theseus."""
    start: float = 120.0
    start_velocity: float = 20.0
    goal_velocity: float = 20.0
    tcp_pos: float = 1200.0
    tcp_rot: float = 180.0
    gp_prior: float = 1.2
    joint_limits: float = 150.0
    link_collision: float = 100.0

@dataclass(frozen=True)
class ErrorScales:
    """Residual normalization scales.

    ``gp_qc`` is the continuous-time white-noise-on-acceleration spectral
    density used by the GPMP2 constant-velocity prior. Larger values make the
    prior softer; smaller values make it stiffer.
    """
    tcp_pos_m: float = 0.01
    tcp_rot_rad: float = 0.05
    joint_rad: float = 0.05
    qdot_rad_s: float = 0.50
    limit_rad: float = 0.05
    limit_margin_rad: float = 0.03
    collision_m: float = 0.03
    gp_dt: float = 0.10
    gp_qc: float = 1.0


NUM_STEPS: int = 32
MAX_ITERATIONS: int = 10
POINTS_PER_LINK: int = 25
TRAJECTORY_DT: float = 0.10

# Default robot position
Q_START: Tuple[float, ...] = (0.0, -0.50, 0.0, -1.60, 0.0, 1.80, 0.0)
Q_REFERENCE_TARGET: Tuple[float, ...] = (1.10, -1.20, 0.50, -2.25, 0.70, 2.50, 1.20)

# Default obstacles used when no named scenario is selected.
OBSTACLE_CENTERS: Tuple[Tuple[float, float, float], ...] = (
    (0.125, 0.225, 0.820),
    (0.250, 0.195, 0.845),
)
OBSTACLE_RADII: Tuple[float, ...] = (0.040, 0.045)
SAFETY_MARGIN: float = 0.002
LINK_COLLISION_RADIUS: float = 0.055

# TCP in Gripper
TCP_LINK_NAME: str = "panda_grasptarget"



@dataclass(frozen=True)
class PlanningScenario:
    name: str
    q_start: Tuple[float, ...]
    q_target: Tuple[float, ...]
    obstacle_centers: Tuple[Tuple[float, float, float], ...]
    obstacle_radii: Tuple[float, ...]
    description: str = ""


# Benchmark scenarios
# All targets are joint-space targets; the TCP goal is computed with FK from q_target.
SCENARIOS = {
    "s1_easy_reach": PlanningScenario(
        name="s1_easy_reach",
        q_start=Q_START,
        q_target=(0.45, -0.85, 0.25, -1.95, 0.20, 2.15, 0.45),
        obstacle_centers=((0.15, -0.35, 0.95),),
        obstacle_radii=(0.035,),
        description="Easy target, obstacle far from the main motion. Sanity check.",
    ),
    "s2_central_obstacle": PlanningScenario(
        name="s2_central_obstacle",
        q_start=Q_START,
        q_target=Q_REFERENCE_TARGET,
        obstacle_centers=((0.125, 0.225, 0.820), (0.250, 0.195, 0.845)),
        obstacle_radii=(0.040, 0.045),
        description="Current v49 scene: obstacles close to the TCP/link path.",
    ),
    "s3_narrow_passage": PlanningScenario(
        name="s3_narrow_passage",
        q_start=Q_START,
        q_target=(0.95, -1.15, 0.65, -2.15, 0.85, 2.35, 1.05),
        obstacle_centers=((0.18, 0.145, 0.820), (0.18, 0.305, 0.820)),
        obstacle_radii=(0.045, 0.045),
        description="Two nearby obstacles create a narrow passage.",
    ),
    "s4_goal_near_obstacle": PlanningScenario(
        name="s4_goal_near_obstacle",
        q_start=Q_START,
        q_target=(1.05, -1.05, 0.35, -2.05, 0.55, 2.30, 1.30),
        obstacle_centers=((0.245, 0.190, 0.830), (0.315, 0.220, 0.835)),
        obstacle_radii=(0.040, 0.035),
        description="Obstacle close to the final TCP region; tests final precision.",
    ),


    "s9_validate": PlanningScenario(
        name="s9_validate",
        q_start=Q_START,
        q_target=(-0.9, -1.05, -0.55, -2.20, -0.70, 2.45, -1.10),
        obstacle_centers=(
            (0.080, -0.180, 0.760),
            (0.220, -0.260, 0.820),
           (0.020, -0.450, 0.855),
        ),
        obstacle_radii=(0.045, 0.050, 0.045),
        description="Validation around-back scene with reachable start and goal.",
    ),

    "s10_validate": PlanningScenario(
        name="s10_validate",
        q_start=Q_START,
        q_target=(-0.85, -1.05, -0.55, -2.20, -0.70, 2.45, -1.10),
            obstacle_centers=(
            (0.080, -0.180, 0.760),  
            (0.220, -0.260, 0.820), 
            (0.020, -0.45, 0.855),   
            (0.285, -0.410, 0.930),   
        ),
        obstacle_radii=(0.045, 0.050, 0.045, 0.045),
        description="Validation; tests DGPMP2 trajectory correction.",
    ),

    "s11_validate_mirror": PlanningScenario(
        name="s11_validate_mirror",
        q_start=Q_START,
        q_target=(0.90, -1.05, 0.55, -2.20, 0.70, 2.45, 1.10),
        obstacle_centers=(
            (0.080, 0.180, 0.760),
            (0.220, 0.260, 0.820),
            (0.020, 0.50, 0.855),
        ),
        obstacle_radii=(0.045, 0.050, 0.045),
        description="Mirrored around-back validation scene.",
    ),



    "s6_high_reach": PlanningScenario(
        name="s6_high_reach",
        q_start=Q_START,
        q_target=(0.15, -0.35, 0.15, -1.10, 0.10, 1.40, 0.35),
        obstacle_centers=((0.20, 0.05, 0.700),),
        obstacle_radii=(0.050,),
        description="Higher TCP target; checks behavior near upper workspace.",
    ),
    "s7_low_reach": PlanningScenario(
        name="s7_low_reach",
        q_start=Q_START,
        q_target=(0.70, -1.55, 0.25, -2.65, 0.25, 2.75, 0.75),
        obstacle_centers=((0.20, 0.15, 0.620), (0.32, 0.10, 0.660)),
        obstacle_radii=(0.045, 0.040),
        description="Lower target; tends to stress lower links and joint limits.",
    ),
    "s8_clutter": PlanningScenario(
        name="s8_clutter",
        q_start=Q_START,
        q_target=(1.15, -1.25, 0.70, -2.30, 0.90, 2.55, 1.35),
        obstacle_centers=(
        (0.26, 0.20, 0.78),
        (0.36, 0.14, 0.78),
        (0.30, 0.28, 0.70),
        (0.44, 0.08, 0.70),
    ),
        obstacle_radii=(0.040, 0.045, 0.035, 0.035),
        description="Several obstacles; hardest benchmark scene.",
    ),
}

DEFAULT_SCENARIO_NAME: str = "s2_central_obstacle"


def get_scenario(name: str) -> PlanningScenario:
    """Return a named scenario or raise a readable error."""
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{name}'. Available scenarios: {available}") from exc
