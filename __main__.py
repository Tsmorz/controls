"""Basic docstring for my module."""

import argparse
import os
from enum import Enum, auto

from loguru import logger

from config.definitions import DEFAULT_DISCRETIZATION
from src.modules.simulator import mass_spring_damper_model


class Pipeline(Enum):
    """Create an enumerator to choose which pipeline to run."""

    KF = auto()
    EKF = auto()
    EKF_SLAM = auto()
    CONTROLLER = auto()
    STATE_SPACE = auto()


def run_state_space_pipeline() -> None:
    """Run the main program with this function."""
    dt = DEFAULT_DISCRETIZATION
    ss_model = mass_spring_damper_model(discretization_dt=dt)
    ss_model.step_response(delta_t=dt, plot_response=True)
    ss_model.impulse_response(delta_t=dt, plot_response=True)


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        action="store",
        type=str,
        required=True,
        help=f"Choose which pipeline to run - {[v.name for v in Pipeline]}",
    )
    args = parser.parse_args()

    pipeline_id = args.pipeline

    if pipeline_id == Pipeline.KF.name:
        os.system("python examples/kf_example.py")
    elif pipeline_id == Pipeline.EKF.name:
        os.system("python examples/ekf_slam_example.py")
    elif pipeline_id == Pipeline.STATE_SPACE.name:
        run_state_space_pipeline()
    else:
        msg = f"Invalid pipeline number: {pipeline_id}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Program complete.")
