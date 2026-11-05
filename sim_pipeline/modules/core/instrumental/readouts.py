"""Write detector readouts to FITS, and extract spectra back out of them."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Sequence

import astropy.units as u
import numpy as np
from astropy.io import fits

from ...pipeline_registry import pipeline_stage
from .channels import FITS_CHANNEL_NAMES
from .detector import SYSTEMATICS_KEY
from .sharp_birchall import SharpBirchallExtractor, rebin_columns_to_bins

if TYPE_CHECKING:
    from .channels import OutputChannel
    from .detector import Detector

logger = logging.getLogger(__name__)


def _readout_cube_adu(output_channel: OutputChannel) -> tuple[list[str], np.ndarray]:
    """Stack a channel's readout frames into
    (n_components, n_readouts, n_pix, n_pix) ADU.
    """
    frames_list = output_channel.readout_frames
    if not frames_list:
        raise RuntimeError(
            f"{output_channel.name} has no readout frames; run "
            "expose_plus_detector_effects() first"
        )
    gain_e_per_adu = output_channel.detector.gain.to_value(u.electron / u.adu)
    components = list(frames_list[0].source_e) + [SYSTEMATICS_KEY]
    cube = np.stack(
        [
            np.stack(
                [
                    (
                        frames.systematics_e
                        if comp == SYSTEMATICS_KEY
                        else frames.source_e[comp]
                    )
                    for frames in frames_list
                ]
            )
            for comp in components
        ]
    )
    return components, cube / gain_e_per_adu


@pipeline_stage(depends_on=("expose_plus_detector_effects",))
def write_detector_readouts(
    output_channels: dict,
    file_name: str,
    t_frame: u.Quantity,
    header_cards: dict | None = None,
    cutout: bool = True,
) -> str:
    """
    Write one FITS file of detector readouts, with one image HDU per output channel.

    Each channel HDU (EXTNAME = CH1BR, CH2BR, CH3DK or CH4DK) holds a cube
    (n_components, n_readouts, ny, nx) in ADU. The components are each source
    and the detector systematics, named in the header as COMP0, COMP1, ...

    With ``cutout=True`` only the bounding box of the spectral trace is written,
    grown by the detector's [detector] buffer_footprint (y, x) pixels on each side
    (clipped to the detector) so some unilluminated pixels are visible; Y0/X0 give
    its lower-left pixel on the full frame (FULLNY x FULLNX), BUFFY/BUFFX the buffer.
    """
    primary = fits.PrimaryHDU()
    primary.header["TFRAME"] = (
        t_frame.to_value(u.s),
        "Integration time of one frame [s]",
    )
    for key, val in (header_cards or {}).items():
        primary.header[key] = val

    hdus = [primary]
    for output_channel in output_channels.values():
        components, cube_adu = _readout_cube_adu(output_channel)
        n_y_full, n_x_full = cube_adu.shape[-2:]
        y0, x0 = 0, 0
        buffer_y, buffer_x = 0, 0
        if cutout:
            det = output_channel.detector
            buffer_y, buffer_x = det.footprint_buffer_yx
            y0, y1, x0, x1 = det.footprint_bbox(buffer_yx=(buffer_y, buffer_x))
            cube_adu = cube_adu[..., y0:y1, x0:x1]
        hdu = fits.ImageHDU(data=cube_adu, name=FITS_CHANNEL_NAMES[output_channel.name])
        hdu.header["BUNIT"] = "adu"
        hdu.header["Y0"] = (y0, "Cutout lower row on the full frame (0-based)")
        hdu.header["X0"] = (x0, "Cutout lower column on the full frame (0-based)")
        hdu.header["FULLNY"] = (n_y_full, "Full-frame rows")
        hdu.header["FULLNX"] = (n_x_full, "Full-frame columns")
        hdu.header["BUFFY"] = (buffer_y, "Cutout rows of buffer around the trace")
        hdu.header["BUFFX"] = (buffer_x, "Cutout columns of buffer around the trace")
        hdu.header["GAIN"] = (
            output_channel.detector.gain.to_value(u.electron / u.adu),
            "Detector gain [e-/ADU]",
        )
        hdu.header["NCOMP"] = (len(components), "Number of signal components (axis 4)")
        hdu.header["NREAD"] = (cube_adu.shape[1], "Number of readouts (axis 3)")
        for i, comp in enumerate(components):
            hdu.header[f"COMP{i}"] = comp
        hdus.append(hdu)

    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(file_name, overwrite=True)
    logger.info("Wrote detector readouts to %s", file_name)
    return file_name


EXTRACTION_METHODS = ("sharp_birchall", "box_sum")


def _model_variance_adu2(
    total_adu: np.ndarray,
    gain_e_per_adu: float,
    dc_rate_e_pix_s: float,
    t_frame_s: float,
    read_noise_e: float,
) -> np.ndarray:
    """
    Per-pixel variance (ADU^2) of one readout: shot noise on the expected charge,
    plus dark charge, plus read noise.

    ## ## TODO: the readout frames are noiseless expectation values, so this is a
    model, not a measurement. Once frames include noise, replace it (e.g. with a
    variance estimated from the noisy frames).
    """
    charge_e = (
        np.clip(total_adu * gain_e_per_adu, 0.0, None) + dc_rate_e_pix_s * t_frame_s
    )
    return charge_e / gain_e_per_adu**2 + (read_noise_e / gain_e_per_adu) ** 2


def _trace_profile_and_fractions(
    det: Detector, y0: int, x0: int, n_y: int, n_x: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    For the cutout box: the trace's spatial profile (1, ny, nx), normalized to sum
    to 1 down each illuminated column, and each column's fractional coverage by
    each wavelength bin (n_bins, nx). Unilluminated columns (e.g. the cutout
    buffer) get a zero profile and zero fractions, so they add nothing.
    """
    fp = det.footprint_cube[:, y0 : y0 + n_y, x0 : x0 + n_x]
    image = fp.sum(axis=0)
    col_sum = image.sum(axis=0)
    profile = np.divide(
        image, col_sum, out=np.zeros_like(image, dtype=float), where=col_sum > 0
    )
    fraction = np.divide(
        fp.sum(axis=1),
        col_sum,
        out=np.zeros((fp.shape[0], n_x), dtype=float),
        where=col_sum > 0,
    )
    return profile[np.newaxis], fraction


def _box_sum_spectra(
    det: Detector, data_adu: np.ndarray, y0: int, x0: int
) -> np.ndarray:
    n_y, n_x = data_adu.shape[-2:]
    bin_map = det.bin_index_map()[y0 : y0 + n_y, x0 : x0 + n_x]
    on_footprint = bin_map >= 0
    # (n_components, n_readouts, n_bins)
    return np.stack(
        [
            np.stack(
                [
                    np.bincount(
                        bin_map[on_footprint],
                        weights=readout_adu[on_footprint],
                        minlength=det.num_wavel_bins,
                    )
                    for readout_adu in component_adu
                ]
            )
            for component_adu in data_adu
        ]
    )


def _sharp_birchall_spectra(
    det: Detector,
    data_adu: np.ndarray,
    y0: int,
    x0: int,
    dc_rates: np.ndarray,
    t_frame_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns flux (n_dc, n_comp, n_read, n_bins), error
    and vark (n_dc, n_read, n_bins).
    """
    n_comp, n_read, n_y, n_x = data_adu.shape
    profile, fraction = _trace_profile_and_fractions(det, y0, x0, n_y, n_x)
    gain = det.gain.to_value(u.electron / u.adu)
    read_noise_values = np.atleast_1d(det.read_noise_e_rms.to_value(u.electron / u.pix))
    if read_noise_values.size > 1:
        logger.warning(
            "Several read-noise values configured (%s); Sharp & Birchall extraction "
            "uses the first, %s e-",
            read_noise_values,
            read_noise_values[0],
        )
    read_noise_e = float(read_noise_values[0]) if read_noise_values.size else 0.0

    n_bins = fraction.shape[0]
    flux = np.zeros((len(dc_rates), n_comp, n_read, n_bins))
    var_flux = np.zeros((len(dc_rates), n_read, n_bins))
    vark = np.zeros((len(dc_rates), n_read, n_bins))
    for i_dc, dc_rate in enumerate(dc_rates):
        for r in range(n_read):
            frames_adu = data_adu[:, r]  # (n_comp, ny, nx)
            variance = _model_variance_adu2(
                frames_adu.sum(axis=0), gain, float(dc_rate), t_frame_s, read_noise_e
            )
            extractor = SharpBirchallExtractor(profile, variance, read_noise_e / gain)
            col_flux, col_var, col_vark = extractor.extract(frames_adu)
            # single trace: spectrum index 0
            flux[i_dc, :, r], var_b = rebin_columns_to_bins(
                fraction, col_flux[:, 0, :], col_var[0]
            )
            var_flux[i_dc, r] = var_b
            # vark rebins like a variance: (fraction**2) @ vark
            vark[i_dc, r] = (fraction**2) @ col_vark[0]
    return flux, np.sqrt(var_flux), vark


@pipeline_stage(depends_on=("write_detector_readouts",))
def extract_spectra_from_readouts(
    output_channels: dict,
    file_name: str,
    *,
    method: str = "sharp_birchall",
    t_frame: u.Quantity | None = None,
    dc_rates: Sequence[float] | np.ndarray | None = None,
) -> Dict[str, Dict[str, u.Quantity]]:
    """
    Read the readout FITS file (cutout or full frame) back in and extract each
    readout of each channel into counts vs. wavelength bin (ADU per frame), once
    per dark-current value.

    method:
        "sharp_birchall"  Sharp & Birchall (2010) optimal extraction per detector
                          column (see ``sharp_birchall.py``), rebinned into wavelength
                          bins by each column's fractional bin coverage. Also gives
                          1-sigma errors (diag C^-1) and the Eqn 19 vark, from a
                          modelled variance image for each DC value.
        "box_sum"         each pixel summed into the one bin covering most of it
                          (``Detector.bin_index_map``); no errors, DC-independent.

    Stores on each channel:
        extracted_spectra_adu[component]  Quantity (n_dc, n_readouts, n_bins), ADU
        extracted_error_adu               Quantity (n_dc, n_readouts, n_bins), ADU, or
        None
        extracted_vark_adu2               Quantity (n_dc, n_readouts, n_bins), ADU^2, or
        None

    Returns: {channel name: {component name: Quantity (n_dc, n_readouts, n_bins)}}
    """
    if method not in EXTRACTION_METHODS:
        raise ValueError(
            f"Unknown extraction method {method!r}; choose from {EXTRACTION_METHODS}"
        )
    dc_rates = np.atleast_1d(
        np.asarray([0.0] if dc_rates is None else dc_rates, dtype=float)
    )
    if method == "sharp_birchall":
        if t_frame is None:
            raise ValueError(
                "Sharp & Birchall extraction needs t_frame for the variance model"
            )
        logger.warning(
            "Sharp & Birchall variances are modelled from noiseless readout frames "
            "(shot noise of the expected charge + dark charge + read noise); update "
            "_model_variance_adu2 in readouts.py once frames include real noise"
        )

    extracted = {}
    with fits.open(file_name) as hdul:
        for output_channel in output_channels.values():
            hdu = hdul[FITS_CHANNEL_NAMES[output_channel.name]]
            components = [hdu.header[f"COMP{i}"] for i in range(hdu.header["NCOMP"])]
            data_adu = np.asarray(hdu.data, dtype=float)  # (n_comp, n_read, ny, nx)
            det = output_channel.detector
            y0, x0 = hdu.header["Y0"], hdu.header["X0"]

            if method == "box_sum":
                box = _box_sum_spectra(
                    det, data_adu, y0, x0
                )  # (n_comp, n_read, n_bins)
                flux = np.broadcast_to(box, (len(dc_rates),) + box.shape)
                output_channel.extracted_error_adu = None
                output_channel.extracted_vark_adu2 = None
            else:
                flux, error, vark = _sharp_birchall_spectra(
                    det, data_adu, y0, x0, dc_rates, t_frame.to_value(u.s)
                )
                output_channel.extracted_error_adu = error * u.adu
                output_channel.extracted_vark_adu2 = vark * u.adu**2

            spectra = {
                comp: np.ascontiguousarray(flux[:, i]) * u.adu
                for i, comp in enumerate(components)
            }
            output_channel.extracted_spectra_adu = spectra
            extracted[output_channel.name] = spectra
    logger.info(
        "Extracted spectra (%s) from detector readouts in %s", method, file_name
    )
    return extracted
