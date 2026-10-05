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
        base["t_int_frame"] = Column(
            data=t_frame,
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
                data=np.sqrt((dc_rate * u.electron / u.pix) * qt["npix"] * t_frame).value
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


@pipeline_stage(depends_on=("combine_astro_and_instrum_signals",))
def chop_signal(output_channels: dict, plot: bool = False) -> dict:
    """
    Subtracts the dark 3 and 4 outputs

    OUTPUTS:
    - dict of chopped tables, keyed by dc_rate
    """

    post_chop_tables_by_dark_current = {}
    for dc_rate, t3 in output_channels["output_3_dark"].tables_by_dark_current.items():
        t1 = output_channels["output_1_bright"].tables_by_dark_current[dc_rate]
        t2 = output_channels["output_2_bright"].tables_by_dark_current[dc_rate]
        t4 = output_channels["output_4_dark"].tables_by_dark_current[dc_rate]
        chopped = QTable()

        # copy wavelength metadata once
        for col in ("bin", "center", "width", "npix"):
            chopped[col] = Column(
                data=t3[col],
                description=t3[col].info.description,
            )

        # keep outputs, but add the chopped signal
        # consolidate signals from dark outputs, and the chopped signal
        for col in t3.colnames:
            if (
                col.startswith("astro_") and col.endswith("_adu")
            ) or col == "instrum_systematics_adu":
                chopped[f"output_1_bright_{col}"] = t1[col]
                chopped[f"output_2_bright_{col}"] = t2[col]
                chopped[f"output_3_dark_{col}"] = t3[col]
                chopped[f"output_4_dark_{col}"] = t4[col]
                chopped[f"chopped_{col}"] = Column(
                    data=t3[col] - t4[col],
                    description=f"Chopped (output_3 − output_4) for {col}",
                )
            if col in ("instrum_dc_rms_adu", "instrum_rn_rms_adu"):
                chopped[f"output_3_dark_{col}"] = t3[col]
                chopped[f"output_4_dark_{col}"] = t4[col]

                ## ## TODO: MAKE SURE THIS IS CORRECT
                chopped[f"chopped_{col}"] = Column(
                    data=np.sqrt(t3[col] ** 2 + t4[col] ** 2),
                    description=f"Combined dark-pair RMS for {col}: sqrt(out3^2 + out4^2)",
                )

        chopped.meta.update(t3.meta)

        post_chop_tables_by_dark_current[dc_rate] = chopped

    return post_chop_tables_by_dark_current
