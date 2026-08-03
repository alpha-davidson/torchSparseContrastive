"""Small helper for saving all currently open Matplotlib figures to one PDF."""

from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_image(filename: str | Path) -> None:
    """Save every open Matplotlib figure into a multipage PDF."""
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for figure_number in plt.get_fignums():
            pdf.savefig(plt.figure(figure_number))
