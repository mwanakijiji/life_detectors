import logging

import numpy as np
import astropy.units as u
from astropy.table import QTable, Column

from ...pipeline_registry import pipeline_stage
from ...utils.helpers.keys import canonical_dc_rate
from ...viz.tables import plot_debug_final_table
from .detector import SYSTEMATICS_KEY
from .dispersion import DispersionLaw


def build_base_astro_table(output_channel, sources_to_include) -> QTable:
    # builds a table that consolidates astrophysical signals, before including instrumental noise
    qt = QTable()
    qt["bin"] = Column(
        data=np.arange(len(output_channel.bin_centers)),
        description="Wavelength bin index (0-based)",
    )
    qt["center"] = Column(
        data=output_channel.bin_centers,
        description="Wavelength bin center",
    )
    qt["width"] = Column(
        data=output_channel.bin_widths,
        description="Wavelength bin width",
    )
    qt.meta["wavel_bin_edges"] = output_channel.bin_edges
    qt["npix"] = Column(
        data=DispersionLaw.n_pix_per_bin(output_channel.detector.footprint_cube),
        description="Number of detector pixels in wavelength bin footprint",
    )
    for source_name in sources_to_include:
        if source_name not in output_channel.astroph_signal:
            continue
        sig = output_channel.astroph_signal[source_name]
        qt[f"astro_{source_name}_ph_s_um"] = Column(
            data=sig["flux_astro_1d_interpolated_ph_sec_um"],
            description=f"{source_name}: photon rate per micron (all pixels in bin)",
        )
        qt[f"astro_{source_name}_ph_s_bin"] = Column(
            data=sig["flux_astro_1d_interpolated_ph_sec_wavel_bin"],
            description=f"{source_name}: photon rate integrated over wavelength bin (all pixels)",
        )
        qt[f"astro_{source_name}_ph_s_pix"] = Column(
            data=sig["flux_astro_1d_interpolated_ph_sec_pixel"],
            description=f"{source_name}: photon rate per pixel (mean over bin footprint)",
        )
    return qt


@pipeline_stage(depends_on=("apply_detector_effects",))
def combine_astro_and_instrum_signals(
    config, output_channels: dict, sources_to_include, plot: bool = False
):
    """
    Combines astrophysical signals and instrumental noise into one table for permutations of
        1) output
        2) dark current
        3) rotation angle

    Astro and systematics signals come from the detector frames
    (``output_channel.detector_signal_e_bin``, set by ``apply_detector_effects``);
    RN / DC / gain come from ``output_channel.detector``.

    OUTPUTS:
    - none; updates output_channel.tables_by_dark_current[canonical_dc_rate(dc_rate)]
    """

    t_frame = float(config["observation"]["t_int_frame"]) * u.second

    # loop over output channels and make a set of tables (one for each value of dark current)
    for output_name, output_channel in output_channels.items():
        det = output_channel.detector
        read_noise_scalar = det.read_noise_e_rms  # just one value here
        gain = det.gain
        signal_e_bin = output_channel.detector_signal_e_bin

        # build table that includes astrophysical signals
        base = build_base_astro_table(output_channel, sources_to_include)
        ## ## TODO: enable multiple reads
        # one value per bin: a scalar Quantity Column loses its name when assigned
        base["t_int_frame"] = Column(
            data=np.full(len(base), t_frame.value) * t_frame.unit,
            description="Integration time of one frame",
        )
        base["qe"] = Column(
            data=det.quantum_efficiency,
            description="Detector quantum efficiency",
        )

        # astrophysical sources → ADU per integration (includes × t_frame); independent of DC
        for source_name in sources_to_include:
            base[f"astro_{source_name}_adu"] = Column(
                data=signal_e_bin[source_name] / gain,
                description=f"{source_name}: signal in wavel bin and detector footprint for one integration (ADU)",
            )
        base["instrum_systematics_adu"] = Column(
            data=signal_e_bin[SYSTEMATICS_KEY] / gain,
            description="Detector systematics (non-astrophysical charge) in wavel bin and detector footprint, for one integration (ADU)",
        )

        # loop over dark current values
        for dc_rate_this in det.dark_current_rate_e_pix_sec:
            dc_rate = canonical_dc_rate(dc_rate_this.value)
            logging.info(
                f"Combining astrophysical signals and instrumental noise for output {output_name} at dark current {dc_rate_this}"
            )

            # one table of signals for a permutation of
            # 1) output
            # 2) dark current
            # 3) rotation angle
            qt = base.copy()

            # 'dark current' pedestal vs RMS: here we store Poisson RMS so dark-subtraction
            # noise can be propagated as if subtraction were already performed
            qt["instrum_dc_rms_adu"] = Column(
                data=np.sqrt(
                    (dc_rate * u.electron / u.pix) * qt["npix"] * t_frame
                ).value
                * u.electron
                / gain,
                description="Dark-current Poisson RMS over wavel bin and detector footprint, for one integration (ADU)",
            )
            qt["instrum_rn_rms_adu"] = Column(
                data=read_noise_scalar * np.sqrt(qt["npix"]).value * u.pix / gain,
                description="Read-noise RMS over wavel bin and detector footprint, for one integration (ADU)",
            )

            # store the final table for this permutation of output, dark current, and rotation angle
            output_channel.tables_by_dark_current[dc_rate] = qt

            # plot of final signal in the detector
            if plot:  # pragma: no cover
                file_name_plot = (
                    str(config["dirs"]["save_s2n_data_unique_dir"])
                    + f"debug_final_table_{output_name}_dc_{dc_rate:.3f}.png"
                )
                plot_debug_final_table(
                    qt,
                    output_channel.bin_edges,
                    output_name,
                    dc_rate,
                    sources_to_include,
                    file_name_plot,
                )
    # for pickling data for Kira
    """
    out = Path("/Users/eckhartspalding/Downloads/kira/output_channels.pkl")
    payload = {
        name: ch.tables_by_dark_current[0.2]
        for name, ch in self.output_channels.items()
    }
    with out.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    ipdb.set_trace()
    """
