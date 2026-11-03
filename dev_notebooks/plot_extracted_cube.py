# Plots an extracted-spectra cube (extracted_cube_qe_{qe}_dc_{dc}.fits) as 2D color maps
# of counts: y = wavelength, x = readout number, one panel per output channel.
#
# Usage:
#   python plot_extracted_cube.py path/to/extracted_cube_qe_0.80_dc_00.200.fits
#   python plot_extracted_cube.py cube.fits --component star --log --output star.png
#
# --component: PRIMARY (total counts, default) or a component HDU name, e.g. star,
#   exoplanet_model_10pc, instrum_systematics, dark_current

import argparse

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits_file", help="extracted_cube_*.fits file")
    parser.add_argument(
        "--component",
        default="PRIMARY",
        help="HDU to plot: PRIMARY (total) or a component name (default: PRIMARY)",
    )
    parser.add_argument("--log", action="store_true", help="log color scale")
    parser.add_argument("--output", help="save figure to this path instead of showing it")
    args = parser.parse_args()

    with fits.open(args.fits_file) as hdul:
        header = hdul[0].header
        counts = np.asarray(hdul[args.component.upper()].data, dtype=float)  # (n_bins, n_readouts, n_channels)
        unit = hdul[args.component.upper()].header.get("BUNIT", "")
        wavelength = hdul["WAVELENGTH"].data
        wavel_edges = np.append(wavelength["lower_edge"], wavelength["upper_edge"][-1])

    n_bins, n_readouts, n_channels = counts.shape
    channel_names = [header.get(f"CHAN{i}", f"channel {i}") for i in range(n_channels)]
    readout_edges = np.arange(n_readouts + 1) - 0.5  # readout numbers 0 .. n-1 at pixel centers

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
    for i_ch, ax in enumerate(axes.ravel()[:n_channels]):
        image = counts[:, :, i_ch]
        norm = None
        if args.log:
            positive = image[image > 0]
            # log scale needs positive values; leave the panel linear if there are none
            if positive.size:
                norm = LogNorm(vmin=positive.min(), vmax=positive.max())
        mesh = ax.pcolormesh(readout_edges, wavel_edges, image, shading="flat", norm=norm)
        fig.colorbar(mesh, ax=ax, label=f"counts [{unit}]")
        ax.set_title(channel_names[i_ch])
        ax.set_xlabel("readout number")
        ax.set_ylabel("wavelength [um]")

    fig.suptitle(
        f"{args.component} — QE {header.get('QE')}, DC {header.get('DC')} e-/pix/s, "
        f"{header.get('NANGLES')} angles × {header.get('NREADANG')} readouts"
    )

    if args.output:
        fig.savefig(args.output)
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
