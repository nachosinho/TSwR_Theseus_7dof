"""Simple PyBullet validation and replay for the student version."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import TCP_LINK_NAME
from .kinematics import fk_urdf_tcp, tool_transform_for_tcp_link
from .robot_model import RobotURDFKinematics


def _require_pybullet():
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pybullet first: pip install pybullet") from exc
    return p, pybullet_data


def _hide_camera_preview_windows(p) -> None:
    """Hide PyBullet camera preview panes while keeping debug sliders visible."""
    for flag_name in (
        "COV_ENABLE_RGB_BUFFER_PREVIEW",
        "COV_ENABLE_DEPTH_BUFFER_PREVIEW",
        "COV_ENABLE_SEGMENTATION_MARK_PREVIEW",
    ):
        flag = getattr(p, flag_name, None)
        if flag is None:
            continue
        try:
            p.configureDebugVisualizer(flag, 0)
        except Exception:
            # Older PyBullet builds may not expose every visualizer option.
            pass


def connect(gui: bool = False):
    p, pybullet_data = _require_pybullet()
    cid = p.connect(p.GUI if gui else p.DIRECT)
    if gui:
        _hide_camera_preview_windows(p)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    return p, cid, Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf"


def load_panda(p, urdf_path: Path):
    return p.loadURDF(str(urdf_path), useFixedBase=True)


def joint_and_link_names(p, robot_id):
    joints = {}
    links = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        joints[info[1].decode()] = j
        links[info[12].decode()] = j
    return joints, links


def reset_arm(p, robot_id, q: Sequence[float]):
    joints, _ = joint_and_link_names(p, robot_id)
    for i, value in enumerate(q, start=1):
        p.resetJointState(robot_id, joints[f"panda_joint{i}"], float(value))
    p.stepSimulation()


def link_pose(p, robot_id, link_name: str):
    _, links = joint_and_link_names(p, robot_id)
    state = p.getLinkState(robot_id, links[link_name], computeForwardKinematics=True)
    pos = np.asarray(state[4], dtype=np.float64)
    quat = state[5]
    R = np.asarray(p.getMatrixFromQuaternion(quat), dtype=np.float64).reshape(3, 3)
    return pos, R


def tcp_pose_pybullet(p, robot_id, tcp_name: str):
    """Pose of the selected TCP frame in PyBullet.

    panda_grasptarget is not a real PyBullet link in every URDF, so we compose
    panda_link8 with the same fixed transform used in the torch model.
    """
    _, links = joint_and_link_names(p, robot_id)
    if tcp_name in links:
        return link_pose(p, robot_id, tcp_name)
    pos8, R8 = link_pose(p, robot_id, "panda_link8")
    T = tool_transform_for_tcp_link(tcp_name, 1, dtype=torch.double, device="cpu")[0].numpy()
    return pos8 + R8 @ T[:3, 3], R8 @ T[:3, :3]


def tcp_pose_torch(q_np, tcp_name: str):
    robot = RobotURDFKinematics.franka_panda(dtype=torch.double, device="cpu")
    q = torch.as_tensor(q_np, dtype=torch.double).reshape(1, 7)
    T = fk_urdf_tcp(
        q,
        robot.joint_origin_xyz.view(1, 7, 3),
        robot.joint_origin_rpy.view(1, 7, 3),
        robot.joint_axis.view(1, 7, 3),
        robot.tcp_fixed_transform.view(1, 4, 4),
        tool_transform_for_tcp_link(tcp_name, 1, dtype=torch.double, device="cpu"),
    )[0].numpy()
    return T[:3, 3], T[:3, :3]


def rotation_error_deg(R_goal, R_current):
    R = R_goal.T @ R_current
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def as_traj(arr):
    q = np.asarray(arr, dtype=np.float64)
    return q[0] if q.ndim == 3 else q


def create_sphere(p, center, radius, rgba=(1.0, 0.2, 0.1, 0.45)):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=float(radius))
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=rgba)
    return p.createMultiBody(0.0, col, vis, basePosition=list(map(float, center)))


def draw_waypoint_markers(p, waypoint_positions):
    """Draw all TCP waypoint targets in the GUI, if a trajectory stores them."""
    waypoint_positions = np.asarray(waypoint_positions, dtype=np.float64).reshape(-1, 3)
    for idx, pos in enumerate(waypoint_positions, start=1):
        radius = 0.016 if idx < len(waypoint_positions) else 0.020
        rgba = (0.0, 0.9, 0.1, 0.85) if idx < len(waypoint_positions) else (0.0, 1.0, 0.0, 0.95)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        p.createMultiBody(0.0, -1, vis, basePosition=pos.tolist())
        p.addUserDebugText(
            f"P{idx}",
            (pos + np.asarray([0.0, 0.0, 0.035])).tolist(),
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.0,
            lifeTime=0,
        )


def closest_obstacle_distance(p, robot_id, spheres, q_traj, safety_margin: float):
    best = None
    for k, q in enumerate(q_traj):
        reset_arm(p, robot_id, q)
        for obs_idx, sphere in enumerate(spheres):
            contacts = p.getClosestPoints(robot_id, sphere, distance=10.0)
            if not contacts:
                continue
            c = min(contacts, key=lambda item: item[8])
            if best is None or c[8] < best[2][8]:
                best = (k, obs_idx, c)
    if best is None:
        return None
    knot, obs_idx, c = best
    dist = float(c[8])
    return {
        "distance": dist,
        "margin": dist - safety_margin,
        "knot": knot,
        "obstacle": obs_idx,
        "link_index": int(c[3]),
        "robot_point": np.asarray(c[5], dtype=np.float64),
        "obstacle_point": np.asarray(c[6], dtype=np.float64),
    }


IGNORED_SELF_PAIRS = {
    tuple(sorted(x))
    for x in [
        ("panda_link7", "panda_link8"),
        ("panda_link7", "panda_hand"),
        ("panda_link8", "panda_hand"),
        ("panda_hand", "panda_leftfinger"),
        ("panda_hand", "panda_rightfinger"),
        ("panda_leftfinger", "panda_rightfinger"),
    ]
}


def link_name(p, robot_id, idx):
    if idx < 0:
        return "base"
    return p.getJointInfo(robot_id, int(idx))[12].decode()


def ignore_self_pair(p, robot_id, a, b):
    a, b = sorted((int(a), int(b)))
    if a == b:
        return True
    if a >= 0 and abs(a - b) <= 1:
        return True
    if a < 0 and b <= 1:
        return True
    names = tuple(sorted((link_name(p, robot_id, a), link_name(p, robot_id, b))))
    if names in IGNORED_SELF_PAIRS:
        return True
    if a >= 7 and b >= 7:
        return True
    return False


def closest_self_distance(p, robot_id, q_traj, query_distance=0.08):
    best = None
    ignored = 0
    for k, q in enumerate(q_traj):
        reset_arm(p, robot_id, q)
        for c in p.getClosestPoints(robot_id, robot_id, distance=query_distance):
            a, b = int(c[3]), int(c[4])
            if ignore_self_pair(p, robot_id, a, b):
                ignored += 1
                continue
            if best is None or c[8] < best[1][8]:
                best = (k, c)
    if best is None:
        return {"distance": float("inf"), "collision": False, "knot": -1, "pair": ("none", "none"), "ignored": ignored}
    k, c = best
    dist = float(c[8])
    return {
        "distance": dist,
        "collision": dist < 0.0,
        "knot": k,
        "pair": (link_name(p, robot_id, int(c[3])), link_name(p, robot_id, int(c[4]))),
        "ignored": ignored,
    }


def draw_tcp_trace(p, robot_id, q_traj, tcp_name, color):
    pts = []
    for q in q_traj:
        reset_arm(p, robot_id, q)
        pos, _ = tcp_pose_pybullet(p, robot_id, tcp_name)
        pts.append(pos)
    for a, b in zip(pts[:-1], pts[1:]):
        p.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=3.0, lifeTime=0)


def manual_frame_slider(p, robot_id, q_traj, *, poll_fps: float = 30.0):
    """Drive the robot with a PyBullet GUI slider over trajectory frames.

    The slider value is continuous in PyBullet, so it is rounded to the nearest
    integer frame. The loop exits when the user presses Ctrl+C in the terminal
    or closes/disconnects the PyBullet GUI.
    """
    q_traj = np.asarray(q_traj, dtype=np.float64)
    if q_traj.ndim != 2 or q_traj.shape[0] == 0:
        raise ValueError(f"Expected a non-empty [frames, dof] trajectory, got shape {q_traj.shape}.")

    import time

    max_frame = int(q_traj.shape[0] - 1)
    frame_slider_id = p.addUserDebugParameter("trajectory frame", 0, max_frame, 0)
    text_id = -1
    last_frame = None
    delay = 1.0 / max(float(poll_fps), 1e-6)

    print(f"Manual frame slider enabled: frames 0..{max_frame}.")
    print("Move the 'trajectory frame' slider in the PyBullet GUI. Press Ctrl+C or close the GUI to stop.")

    try:
        while True:
            raw_frame = p.readUserDebugParameter(frame_slider_id)
            frame = int(np.clip(np.rint(raw_frame), 0, max_frame))
            if frame != last_frame:
                reset_arm(p, robot_id, q_traj[frame])
                if text_id >= 0:
                    try:
                        p.removeUserDebugItem(text_id)
                    except Exception:
                        pass
                text_id = p.addUserDebugText(
                    f"frame {frame}/{max_frame}",
                    [0.05, -0.55, 1.15],
                    textColorRGB=[1.0, 1.0, 1.0],
                    textSize=1.4,
                    lifeTime=0,
                )
                last_frame = frame
            time.sleep(delay)
    except KeyboardInterrupt:
        print("Manual frame slider interrupted by user.")
    except Exception as exc:
        # Closing the GUI commonly raises a PyBullet connection/read error.
        print(f"Manual frame slider stopped: {exc}")


def validate_and_replay(path: str | Path, *, gui=False, replay=False, hold=False, loops: int = 1, loop_forever: bool = False, replay_fps: float = 30.0, slider: bool = False, slider_fps: float = 30.0):
    data = np.load(path, allow_pickle=True)
    q_init = as_traj(data["q_init"])
    q_solution = as_traj(data["q_solution"])
    tcp_name = str(np.asarray(data.get("tcp_link_name", [TCP_LINK_NAME])).reshape(-1)[0])
    centers = np.asarray(data["obstacle_centers"], dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(data["obstacle_radii"], dtype=np.float64).reshape(-1)
    safety = float(np.asarray(data["safety_margin"]).reshape(-1)[0])

    if slider and not gui:
        print("Manual frame slider requires PyBullet GUI, enabling gui=True.")
        gui = True

    p, cid, urdf = connect(gui=gui)
    try:
        robot = load_panda(p, urdf)
        print("=== PyBullet validator ===")
        print(f"Panda URDF: {urdf}")
        print(f"TCP frame: {tcp_name}")

        for label, q in [("q_start", q_solution[0]), ("q_final", q_solution[-1])]:
            reset_arm(p, robot, q)
            torch_pos, torch_R = tcp_pose_torch(q, tcp_name)
            pb_pos, pb_R = tcp_pose_pybullet(p, robot, tcp_name)
            print(f"FK {label}: pos error = {np.linalg.norm(torch_pos - pb_pos):.9f} m, rot error = {rotation_error_deg(torch_R, pb_R):.9f} deg")

        spheres = [create_sphere(p, c, r) for c, r in zip(centers, radii)]
        result = closest_obstacle_distance(p, robot, spheres, q_solution, safety)
        self_result = closest_self_distance(p, robot, q_solution)

        goal_pos = np.asarray(data["tcp_goal_pos"], dtype=np.float64).reshape(-1, 3)[0]
        goal_R = np.asarray(data["tcp_goal_rot"], dtype=np.float64).reshape(-1, 3, 3)[0]
        waypoint_goal_pos = None
        if "waypoint_tcp_goal_pos" in data:
            waypoint_goal_pos = np.asarray(data["waypoint_tcp_goal_pos"], dtype=np.float64).reshape(-1, 3)
            print(f"waypoint targets stored:      {len(waypoint_goal_pos)}")
        reset_arm(p, robot, q_solution[-1])
        final_pos, final_R = tcp_pose_pybullet(p, robot, tcp_name)
        print(f"final TCP position error [m]: {np.linalg.norm(final_pos - goal_pos):.9f}")
        print(f"final TCP rotation error [deg]: {rotation_error_deg(goal_R, final_R):.9f}")
        if result:
            print(f"obstacle min distance [m]:   {result['distance']:.5f}")
            print(f"obstacle safety margin [m]:  {result['margin']:.5f}")
            print(f"obstacle min knot:           {result['knot']}")
            print(f"closest obstacle index:      {result['obstacle']}")
            print(f"closest robot link:          {link_name(p, robot, result['link_index'])}")
            print(f"physical collision:          {result['distance'] < 0.0}")
            print(f"clearance satisfied:         {result['margin'] > 0.0}")
        print(f"self-collision free:         {not self_result['collision']}")
        print(f"self min distance [m]:       {self_result['distance']:.5f}")
        print(f"self closest links:          {self_result['pair'][0]} <-> {self_result['pair'][1]}")
        print(f"ignored self contacts:       {self_result['ignored']}")

        if gui:
            # Goal marker(s) and TCP traces.
            if waypoint_goal_pos is not None:
                draw_waypoint_markers(p, waypoint_goal_pos)
            else:
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.018, rgbaColor=(0.0, 1.0, 0.0, 0.9))
                p.createMultiBody(0.0, -1, vis, basePosition=goal_pos.tolist())
            draw_tcp_trace(p, robot, q_init, tcp_name, (0.0, 0.0, 1.0))
            draw_tcp_trace(p, robot, q_solution, tcp_name, (1.0, 0.5, 0.0))

        if slider:
            manual_frame_slider(p, robot, q_solution, poll_fps=slider_fps)
            replay = False
            hold = False

        if replay:
            import time

            delay = 1.0 / max(float(replay_fps), 1e-6)
            loops = max(1, int(loops))
            loop_idx = 0
            try:
                while True:
                    for q in q_solution:
                        reset_arm(p, robot, q)
                        time.sleep(delay)
                    loop_idx += 1
                    if not loop_forever and loop_idx >= loops:
                        break
                    time.sleep(0.25)
            except KeyboardInterrupt:
                print("Replay interrupted by user.")
        if gui and hold:
            input("Press Enter to close PyBullet...")
    finally:
        try:
            p.disconnect(cid)
        except Exception:
            # The GUI might already be gone if the user closed the PyBullet window.
            pass
