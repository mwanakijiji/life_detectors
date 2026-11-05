"""Visualization helpers: pure plot functions for pipeline FYI figures."""

from astropy.visualization import quantity_support


def enable_plot_units() -> None:
    """Let matplotlib label axes from astropy Quantity units."""
    quantity_support()
