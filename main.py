"""Basic docstring for my module."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

FIG_SIZE = (8, 7)
NUM_STEPS = 100


def matplotlib_plotter(show_plots: bool = False) -> None:
    """Boilerplate matplotlib plotter.

    :param show_plots: whether to show plots
    :return: None
    """
    fig, ax = plt.subplots(3, 1, figsize=FIG_SIZE)
    time = np.arange(NUM_STEPS)
    data = np.cos(time)
    plt.plot(time, data)
    plt.show(show_plots)
    plt.close()


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        action="store",
        default=None,
        help="Directory to process.",
    )
