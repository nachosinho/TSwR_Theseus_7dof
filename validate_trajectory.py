"""Validate and optionally replay the saved trajectory in PyBullet."""

from __future__ import annotations

import argparse
from pathlib import Path

from franka_dgpmp2.pybullet_validator import validate_and_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", default="last_solution_trajectory.npz")
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI")
    parser.add_argument("--replay", action="store_true", help="Replay q_solution")
    parser.add_argument("--hold", action="store_true", help="Wait for Enter before closing the GUI")
    parser.add_argument("--loops", type=int, default=1, help="Number of replay loops when --replay is used")
    parser.add_argument("--loop-forever", action="store_true", help="Replay continuously until Ctrl+C or window close")
    parser.add_argument("--replay-fps", type=float, default=30.0, help="Replay speed in frames per second")
    parser.add_argument("--slider", action="store_true", help="Open a PyBullet GUI slider for manual frame-by-frame trajectory scrubbing")
    parser.add_argument("--slider-fps", type=float, default=30.0, help="Polling rate for the manual frame slider")
    args = parser.parse_args()

    path = Path(args.trajectory)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}. Run run_planner.py first.")
    validate_and_replay(
        path,
        gui=args.gui,
        replay=args.replay,
        hold=args.hold,
        loops=args.loops,
        loop_forever=args.loop_forever,
        replay_fps=args.replay_fps,
        slider=args.slider,
        slider_fps=args.slider_fps,
    )


if __name__ == "__main__":
    main()
