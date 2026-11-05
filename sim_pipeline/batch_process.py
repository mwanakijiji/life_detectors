#!/usr/bin/env python3
"""
Batch processing script for Life Detectors package.

This script allows you to run multiple calculations with different n_int values
and save the results as FITS files with different names.

Comment convention at each call below:
    # ( what the call does / what it returns or modifies )
    # FYI out: files, plots and log lines it writes, for checking that step
    #          ({save_dir} = [dirs] save_s2n_data_unique_dir of the temp config;
    #          "plot only" = written only if [tasks] print_fyi_plots / plot=True)
"""

import configparser
import copy
import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Optional

import astropy.units as u
import numpy as np
import pandas as pd
import yaml
import ipdb

from modules.core import astrophysical
from modules.core.calculator.extracted_cube import assemble_extracted_spectra_cubes

# S/N is disabled for now; kept importable (tests patch it) until it is rebuilt
# pylint: disable-next=unused-import
from modules.core.calculator.snr_from_hdf5 import (  # noqa: F401
    calculate_s2n_post_rotation,
)
from modules.core.instrumental.channels import FITS_CHANNEL_NAMES
from modules.core.instrumental.pipeline import InstrumentDepTerms
from modules.core.planet_spectrum_provider import (
    BlackbodyFromParamsGenerator,
    PopulationSpectrumProvider,
    PSGSpectrumGenerator,
    SingleModelFileProvider,
)
from modules.utils import loader, validator
from modules.utils.helpers.config_ops import (
    _normalize_output_root,
    apply_output_root_override,
    apply_smoke_test_overrides,
    get_sweep_range,
    modify_config_file_pl_system_params,
    modify_config_file_sweep,
    smoke_test_enabled,
    smoke_test_limits,
)
from modules.utils.helpers.formatting import ensure_plot_title_context
from modules.utils.helpers.hdf5_io import record_info_at_angle_and_qe
from modules.utils.helpers.keys import (
    format_angle_qe_fits_readouts_name,
    format_qe_fits_full_example_name,
)
from modules.utils.helpers.spectra import (
    compute_collecting_area_m2,
    create_sample_data,
    merge_psg_spectra_to_planet_population,
)
from modules.viz.spectra import plot_planet_population_sample

# Module-level logger so it's available everywhere in this file
logger = logging.getLogger(__name__)


def run_single_calculation(
    config_path: str,
    base_filename: str,
    sources_to_include: List[str],
    qe: float,
    override_stellar_mask: bool = True,
    overwrite: bool = True,
    plot: bool = False,
    system_params: Optional[dict | pd.Series] = None,
    lum_types: Optional[dict] = None,
    output_root: Optional[str] = None,
    smoke_test: bool = False,
) -> bool:
    """
    Run a single calculation with specified parameters. (Single here means single QE
    value.)

    Args:
        config_path: Path to the configuration file
        base_filename: string to distinguish individual planets (index number from df of
            planet population data)
        sources_to_include: List of sources to include in calculation
        n_int: Number of integrations
        qe: quantum efficiency
        overwrite: Whether to overwrite existing files
        plot: Whether to generate plots
        system_params: Optional[dict] = None: the planetary system parameters
        lum_types: Optional dictionary mapping the luminosities to the stellar types
        smoke_test: force smoke-test mode (small, fast run; see [smoke_test] in the
            config), even if [smoke_test] enabled = False

    Returns:
        output_path: Path of the output FITS file
    """

    # write a log file even when not called via batch_run.py (no-op if one is set up)
    # ( adds a file handler to the root logger if none writes to a real file )
    # FYI out: sim_pipeline/logs/detectorsim_<timestamp>_<tag>.log (only if new);
    #          log line "Log file: ..."
    loader.ensure_logging()

    # get config things sorted out
    base_config = configparser.ConfigParser()
    base_config.read(config_path)
    # ( read True/False flags from [tasks]; nothing written )
    generate_sims = loader.config_getboolean(base_config, "tasks", "generate_sims")
    calculate_s2n = loader.config_getboolean(
        base_config, "tasks", "calculate_s2n_post_rotation"
    )
    # config-driven FYI plotting: overrides the passed-in plot flag when the key is
    # present
    plot = loader.config_getboolean(
        base_config, "tasks", "print_fyi_plots", default=plot
    )
    # ( True if --smoke was passed or [smoke_test] enabled = True; nothing written )
    smoke_test = smoke_test_enabled(base_config, force=smoke_test)

    # Build a per-run token so parallel jobs do not clobber temp files.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_pid{os.getpid()}_{uuid.uuid4().hex[:8]}"

    # Create temporary config file with modified values: use new n_int, QE values
    # TO DO: CHECK THIS FOR CASE WHEN PLANET POPULATION IS NOT BEING DONE; DOES THE
    # ABSENCE OF A BASE_FILENAME CAUSE PROBLEMS?
    # ( copies the base config with [detector] quantum_efficiency = qe; returns the
    #   copy's path )
    # FYI out: <config dir>/parameter_sweeps/<config name>_temp_qe<qe>_<run_id>.ini
    temp_config_path_nint_qe = modify_config_file_sweep(config_path, qe, run_id=run_id)
    # ( if output_root is given, rewrites the [dirs] output root in that temp config
    #   in place; returns the same path )
    # FYI out: log line "Using overridden output root: ..." (only if output_root)
    temp_config_path_nint_qe = apply_output_root_override(
        temp_config_path_nint_qe, output_root
    )
    # smoke-test mode: shrink the run via the [smoke_test] overrides (before the
    # per-planet config, which is derived from this one)
    if smoke_test:
        # ( applies every [smoke_test] "section.key = value" to the temp config in
        #   place )
        # FYI out: WARNING log "*** SMOKE TEST MODE *** ... overrides applied to ..."
        #          listing each override
        apply_smoke_test_overrides(temp_config_path_nint_qe, force=True)
    # modify again, for a given planetary system
    if system_params is not None:
        # ( writes a further temp config with this planet's [target] parameters (Rp,
        #   Ts, Ds, Stype, PSG spectrum file, ...); returns its path )
        # FYI out: {save_dir}<planet/system string>_<run_id>.ini;
        #          log lines "NASA PSG spectrum file name: ..." and
        #          "Created temporary config file for one planetary system: ..."
        temp_config_path = modify_config_file_pl_system_params(
            config_path=temp_config_path_nint_qe,
            base_filename=base_filename,
            system_params=system_params,
            lum_types=lum_types,
            run_id=run_id,
        )
    else:
        temp_config_path = temp_config_path_nint_qe

    # First log entry: which config file we're using (original and temp)
    logger.info("--------------------------------")
    logger.info("Config path (base): %s", config_path)
    logger.info(
        "Config path (temporary one for this case, for batch processing): %s",
        temp_config_path,
    )

    # Load config in two forms:
    # - dict (for validation + directory creation)
    # - ConfigParser (for downstream code that expects .has_section/.options)
    # this is necessary for vestigial reasons while using only one load_config()
    # function
    # ( parses the temp config into a dict (numbers -> float, True/False -> bool) and
    #   creates every directory listed in [dirs] )
    # FYI out: log lines "[section]" / "  key = value" for the whole config, and
    #          "Created directory: ..." per [dirs] entry
    config_dict = loader.load_config(config_file=temp_config_path, makedirs=True)
    # ( checks required sections/fields and value ranges; returns True/False, does
    #   not raise except for non-numeric values )
    # FYI out: ERROR log lines e.g. "Missing telescope field: ..." for each problem
    validator.validate_config(config_dict)
    config = configparser.ConfigParser()
    config.read(temp_config_path)
    # ( builds the plot-title string (system + observation/detector parameters) and
    #   caches it in config; nothing written )
    ensure_plot_title_context(config)

    # in smoke-test mode the overridden temp config decides plotting and run size
    # ( dict: max_qe_values, max_planets, write_full_frame_example; no-op defaults
    #   when smoke mode is off; nothing written )
    run_limits = smoke_test_limits(config, force=smoke_test)
    if smoke_test:
        plot = loader.config_getboolean(
            config, "tasks", "print_fyi_plots", default=plot
        )

    # Make the overwriteable scratch FITS file unique per run.
    if config.has_section("saving") and config.has_option(
        "saving", "save_s2n_data_temp"
    ):
        temp_fits_path = config.get("saving", "save_s2n_data_temp")
        temp_root, temp_ext = os.path.splitext(temp_fits_path)
        config.set("saving", "save_s2n_data_temp", f"{temp_root}_{run_id}{temp_ext}")

    # should we simulate the observations and generate the HDF5 files?
    if generate_sims:

        logger.info("Generating simulations.")

        # S/N results will be written to this file
        # Insert QE and n_int into the filename before .fits
        fname_base = temp_config_path.replace(".ini", "")
        output_fits_file_abs_path = f"{fname_base}_qe_{qe:.4f}_{run_id}.fits"

        # Useful debug info about the loaded config: list actual INI-style sections and
        # key-value pairs
        logger.info("------- Temp config contents -------")

        # log stuff from the config file
        if isinstance(config, configparser.ConfigParser):
            # Config is a ConfigParser: iterate its sections and items
            for section_name in config.sections():
                logger.info("[%s]", section_name)
                for key, value in config[section_name].items():
                    logger.info("  %s = %s", key, value)
        elif isinstance(config, dict):
            # Config is a dict-of-dicts (as returned by some loaders)
            logger.info("Top-level keys: %s", list(config.keys()))
            for section_name, section_dict in config.items():
                logger.info("[%s]", section_name)
                if isinstance(section_dict, dict):
                    for key, value in section_dict.items():
                        logger.info("  %s = %s", key, value)
                else:
                    logger.info("  (section value) %s", section_dict)
        else:
            # Fallback: just log the raw object
            logger.info("(Unrecognized config type %s; raw repr follows)", type(config))
            logger.info(repr(config))
        logger.info(
            "Running calculation with output file name = %s", output_fits_file_abs_path
        )

        # Generate sample spectral data
        logger.info("Creating sample spectral data...")
        # ( generates star BB, planet BB, exozodi and zodi spectra on a 0.1-25 um grid,
        #   writes them as text tables, and points [astrophysical_sources_library] in
        #   config at them (config modified in place) )
        # FYI out: {save_dir}star_spectrum.txt, exoplanet_bb_spectrum.txt,
        #          exozodiacal_spectrum.txt, zodiacal_spectrum.txt;
        #          log line "Created sample data: ..." per file;
        #          plot only: {save_dir}/star_spectrum.png, planet_spectrum.png,
        #          exozodiacal_spectrum.png, zodiacal_spectrum_los.png,
        #          zodiacal_emission_2d.png
        create_sample_data(config, overwrite=overwrite, plot=plot)

        # instantiate astrophysical flux calculator
        logger.info("Calculating astrophysical flux...")
        # planet spectrum source is driven by systems_2_look_at, not by which
        # astrophysical sources happen to be enabled (keeps a single switch)
        systems_2_look_at = config.get(
            "system_options", "systems_2_look_at", fallback="planet_population"
        )
        if systems_2_look_at == "read_in_one_planet_spectrum":
            # ( planet spectrum read from the one model file in the library; nothing
            #   written )
            provider = SingleModelFileProvider(
                config["astrophysical_sources_library"]["exoplanet_model_10pc"]
            )
        else:
            wavelength_range = (
                float(config["wavelength_range"]["min"]),
                float(config["wavelength_range"]["max"]),
            )
            spectrum_method = config.get(
                "planet_population", "spectrum_method", fallback="blackbody"
            )
            if spectrum_method == "blackbody":
                # ( planet spectrum = blackbody from each population row's Tp, Rp;
                #   nothing written )
                generator = BlackbodyFromParamsGenerator(
                    wavelength_range=wavelength_range
                )
            elif spectrum_method == "psg":
                # ( planet spectrum from NASA PSG; nothing written here )
                generator = PSGSpectrumGenerator()
            else:
                raise ValueError(
                    f"Unknown [planet_population] spectrum_method: {spectrum_method!r}"
                )
            # ( wraps the generator so each planet gets its own spectrum )
            provider = PopulationSpectrumProvider(generator)
        # ( holds config + planet provider; computes source fluxes and the sky scene
        #   below; nothing written )
        astrophysical_sources = astrophysical.AstrophysicalSources(
            config, planet_spectrum_provider=provider
        )

        # Calculate incident flux for each source, and add on 2D positions as projected
        # on sky
        sources_astroph = {}
        for source_name in sources_to_include:
            if source_name in [
                "star",
                "star_psg",
                "exoplanet_bb",
                "exoplanet_bb_psg",
                "exoplanet_model_10pc",
                "exoplanet_psg",
                "exozodiacal",
                "exozodiacal_psg",
                "zodiacal",
            ]:
                # ( returns {"wavel", "pre_screen_astro_flux_ph_sec_m2_um"} for this
                #   source, i.e. flux incident at the telescope, ph/s/m^2/um )
                # FYI out: WARNING "Spectrum not available for ..." if unknown;
                #          plot only: {save_dir}incident_<source_name>.png
                sources_astroph[source_name] = (
                    astrophysical_sources.calculate_received_astro_fluxes(
                        source_name=source_name, plot=plot, system_params=system_params
                    )
                )

        ipdb.set_trace()

        # put all the objects into the scene
        logger.info("Generating on-sky scene...")
        # ( returns {source_name: cube (n_wavel, n_pix, n_pix)}: each source placed on
        #   the sky grid ([onsky_scene] n_pix, pix_size_mas, positions); only star,
        #   exoplanet_model_10pc, exozodiacal and zodiacal have scene models )
        # FYI out: log "Constructing scene. <source> flux units are consistent: ...";
        #          WARNING "No on-sky scene model for source ..." for any other source;
        #          plot only: {save_dir}/scene_no_screen_fyi.png and
        #          {save_dir}scene_no_screen.fits (PRIMARY = sum of all sources, one
        #          HDU per source named e.g. STAR, ZODIACAL, then YY_ARCSEC,
        #          XX_ARCSEC, WAVEL_UM); log "Saved on-sky scene (total + ...) to ..."
        astro_scene_perfect_no_screen = astrophysical_sources.arrange_onsky_scene(
            incident_dict=sources_astroph, plot=plot
        )

        ipdb.set_trace()

        # instantiate instrument effects, OutputChannel objects
        logger.info("Passing astrophysical flux through telescope aperture...")
        # ( holds the instrument-side state: sources_astroph, prop_dict, the four
        #   OutputChannels (with geometric wavelength bins from [detector] spec_res),
        #   detector effects ([detector_systematics] enabled) and instrument
        #   backgrounds ([instrument_backgrounds] enabled) )
        # FYI out: "Loaded <effect> map from ..." per file-based effect; WARNING if
        #          read noise / dark current are listed as systematics
        instrument_dep_terms = InstrumentDepTerms(
            config = config,
            sources_astroph = sources_astroph,
            sources_to_include = sources_to_include,
        )

        ipdb.set_trace()

        # pristine copy of incident astrophysical source spectra, before any screen/aperture steps
        sources_astroph_pristine = copy.deepcopy(instrument_dep_terms.sources_astroph)

        # make 1 full rotation
        angles_deg = np.linspace(
            0, 360, num=int(config["observation"]["N_angles"]), endpoint=False
        )
        # loop over angles
        for i_angle, angle_deg in enumerate(angles_deg):
            # Reset mutable pipeline state
            instrument_dep_terms.sources_astroph = copy.deepcopy(
                sources_astroph_pristine
            )
            instrument_dep_terms.prop_dict = (
                {}
            )  # dict to hold astrophysical terms as propagated through the instrument

            # reset the OutputChannel objects for this new angle
            for ch in instrument_dep_terms.output_channels.values():
                ch.angle_deg = angle_deg
                ch.astroph_signal.clear()
                ch.readout_frames.clear()
                ch.extracted_spectra_adu.clear()
                ch.extracted_error_adu = None
                ch.extracted_vark_adu2 = None
            for effect in instrument_dep_terms.detector_effects:
                # ( clears per-angle effect state, e.g. persistence; nothing written )
                effect.reset()

            ipdb.set_trace()

            # generate the transmission screens specific to this angle (one per output)
            logger.info("Generating transmission screens...")
            # ( returns numpy array with slices (ch1, ch2, ch3, ch4, grad_y, grad_x) )
            # ( screens from the aperture-array YAML phases, normalised, rotated by
            #   angle_deg; dark outputs get a small manual stellar mask if
            #   override_stellar_mask )
            # FYI out: log "Normalized transmission instrument response ...",
            #          "Star is manually being nulled to ... on output_3/4_dark";
            #          plot only:
            #          {save_dir}transmission_instrument_response_angle_<angle>.fits
            #          (HDUs OUTPUT_1_BRIGHT, OUTPUT_2_BRIGHT, OUTPUT_3_DARK,
            #          OUTPUT_4_DARK, YY_ARCSEC, XX_ARCSEC, each with QUANTITY and
            #          BUNIT; PRIMARY header ANGLE, WAVEL_M, NORMALIZ, STARMASK) and
            #          {save_dir}differential_dark.fits (output 3 - output 4);
            #          log "Saved transmission instrument responses for angle ..."
            transmission_screens = (
                instrument_dep_terms.generate_instrument_transmission(
                    override_stellar_mask = override_stellar_mask,
                    angle_deg = angle_deg,
                    plot = plot,
                )
            )

            ipdb.set_trace()

            # pass astrophysical scene through transmission screens
            logger.info("Passing astrophysical flux through transmission screens ...")
            # ( updates instrument_dep_terms.sources_astroph in place )
            # ( adds per source, per output: "flux_cube_post_screen_ph_sec_m2_um"
            #   (scene x screen) and "flux_integrated_post_screen_ph_sec_m2_um"
            #   (summed over the sky) )
            # FYI out: log "Flux of <source> passed through transmission screen <out>";
            #          plot only, per source:
            #          {save_dir}flux_of_<source>_passed_through_transmission_screens
            #          _angle_<angle>.png and
            #          {save_dir}source_transmission_map_triptych_<source>_angle_<angle>
            #          _output_<out>.png (x4, wavelength slice 15); plus
            #          {save_dir}photoelectrons_all_sources_pre_aperture.png
            instrument_dep_terms.pass_through_transmission_screens(
                fyi_angle = angle_deg,
                source_dict_pre_screen = astro_scene_perfect_no_screen,
                transmission_screens = transmission_screens,
                plot = plot,
            )

            ipdb.set_trace()

            # Pass through telescope aperture
            # output units: photons/sec/etc.
            logger.info(
                "Passing through telescope aperture (incl. telescope throughput)..."
            )
            # ( fills instrument_dep_terms.prop_dict[source] with
            #   "flux_cube_post_screen_post_aperture_ph_sec_um" = eta_t x collecting
            #   area x post-screen cube, per output (ph/s/m^2/um -> ph/s/um) )
            # FYI out: log "Passed astrophysical flux through telescope aperture...";
            #          plot only, per output:
            #          {save_dir}flux_all_sources_post_screen_pre_aperture_<out>.png,
            #          {save_dir}flux_all_sources_post_screen_post_aperture_<out>.png,
            #          {save_dir}poster_post_aperture_flux_<out>.pdf
            instrument_dep_terms.pass_through_aperture(plot=plot)

            ipdb.set_trace()

            # Pass from aperture to detector
            logger.info("Passing from aperture to detector...")
            # ( adds each enabled instrument background as its own prop_dict entry,
            #   spread uniformly over the sky plane (telescope_thermal is zeros for
            #   now) )
            # FYI out: log "Added instrument background '<name>' into post-aperture
            #          beam"; WARNING if prop_dict is empty;
            #          plot only: {save_dir}beam_to_detector_angle_<angle>.png (one
            #          panel per output: sky-summed ph/s/um of each source (solid),
            #          each instrument background (dashed) and the total, plus the
            #          band-integrated background fraction of the total)
            instrument_dep_terms.pass_from_aperture_to_detector(plot=plot)

            ipdb.set_trace()

            # disperse astrophysical signals on the detector (i.e., update the
            # OutputChannel objects; note that input astrophysical signals should still
            # be photons, not electrons)
            # output units:
            #   ph/sec/um
            #   ph/sec/wavel_bin
            #   ph/sec/pix
            logger.info("Dispersing signals on channel detectors ...")
            # ( per channel: creates the Detector if needed, sets its trace footprint,
            #   sums each source over the sky, rebins onto the channel's wavelength
            #   bins, applies QE, and stores channel.astroph_signal[source] (per um,
            #   per bin, per pixel, n_pix_per_wavel_bin) )
            # FYI out: log "Dispersing signals on detector for <channel>",
            #          "Number of frames at this angle position: ...";
            #          plot only, per channel: {save_dir}footprint_bool_<channel>.png,
            #          {save_dir}astro_ph_sec_pixel_<channel>.csv (ph/s/pix per source
            #          + cumulative) and {save_dir}astro_ph_sec_pixel_<channel>.pdf
            instrument_dep_terms.disperse_astro_signals_on_detector(plot=plot)

            ipdb.set_trace()

            # expose 2D detector frames per readout and apply detector effects
            logger.info("Applying detector effects to readout frames ...")
            # ( per channel and readout ([observation] N_int_per_angle): exposes the
            #   detector for t_int_frame (DetectorFrames: source_e, systematics_e,
            #   total_e) and runs each effect; stores channel.readout_frames )
            # FYI out: log "Applied <n> detector effect(s) over <n> readout(s) for
            #          <channel>"
            instrument_dep_terms.apply_detector_effects()

            # write the detector readouts to FITS (one HDU per output channel):
            # a cutout around the spectral trace at every angle, plus one full-frame
            # example (unless switched off, e.g. in smoke-test mode)
            save_dir = str(config["dirs"]["save_s2n_data_unique_dir"])
            header_cards = {"ANGLE": float(angle_deg), "QE": float(qe)}
            if i_angle == 0 and run_limits["write_full_frame_example"]:
                full_example_file_name = save_dir + format_qe_fits_full_example_name(qe)
                logger.info(
                    "Writing full-frame example readouts to %s ...",
                    full_example_file_name,
                )
                # ( writes the full detector frames (first angle only); HDUs CH1BR,
                #   CH2BR, CH3DK, CH4DK, each (n_components, n_readouts, ny, nx) in
                #   ADU; header COMP0.., GAIN, TFRAME, ANGLE, QE )
                # FYI out: {save_dir}readouts_full_example_qe_<qe>.fits;
                #          log "Wrote detector readouts to ..." (+ size, below)
                instrument_dep_terms.write_detector_readouts(
                    full_example_file_name, header_cards=header_cards, cutout=False
                )
                size_str = (
                    f"{os.path.getsize(full_example_file_name) / 1e6:.1f} MB"
                    if os.path.exists(full_example_file_name)
                    else "size unknown"
                )
                logger.info(
                    "Full-frame example readouts (angle %.2f deg; one HDU per channel: "
                    "%s) "
                    "written to %s (%s)",
                    angle_deg,
                    ", ".join(FITS_CHANNEL_NAMES.values()),
                    os.path.abspath(full_example_file_name),
                    size_str,
                )
            readouts_file_name = save_dir + format_angle_qe_fits_readouts_name(
                angle_deg, qe
            )
            logger.info(
                "Writing detector readout cutouts to %s ...", readouts_file_name
            )
            # ( same as above, but only the trace's bounding box (header Y0, X0,
            #   FULLNY, FULLNX); remembered as the file the extraction reads )
            # FYI out: {save_dir}readouts_angle_<angle>_qe_<qe>.fits;
            #          log "Wrote detector readouts to ..."
            instrument_dep_terms.write_detector_readouts(
                readouts_file_name, header_cards=header_cards, cutout=True
            )

            ipdb.set_trace()

            # extract counts vs. wavelength from the readouts ([extraction] method:
            # Sharp & Birchall or box sum)
            logger.info("Extracting spectra from detector readouts ...")
            # ( reads the cutout FITS back and extracts, per channel and component,
            #   spectra of shape (n_dc, n_readouts, n_bins) in ADU, one slice per
            #   [detector] dark-current value; sets extracted_spectra_adu, and (S&B
            #   only) extracted_errors_adu = {"error", "vark"} )
            # FYI out: log "Extracted spectra (<method>) from detector readouts in ...";
            #          WARNING that S&B variances are modelled from noiseless frames;
            #          WARNING if several read-noise values; WARNING "Sharp & Birchall
            #          solve failed in column ..." on singular columns
            instrument_dep_terms.extract_spectra_from_readouts()

            ipdb.set_trace()

            # write condensed information at this transmission screen angle (to avoid
            # mem leak)
            # ( writes the extracted spectra for this angle: group __extracted_qe_<qe>
            #   with dc_rates, bin geometry (npix, center, width, edges),
            #   <channel>/<component> (n_dc, n_read, n_bins), __error_total,
            #   __vark_total )
            # FYI out: {save_dir}angle_<angle>_qe_<qe>.hdf5;
            #          log "Recorded angle ...: extracted spectra in <file>:<group>"
            record_info_at_angle_and_qe(
                angle_deg=angle_deg,
                qe=qe,
                output_channels=instrument_dep_terms.output_channels,
                save_dir=save_dir,
                t_frame=float(config["observation"]["t_int_frame"]) * u.second,
                extracted_spectra_adu=instrument_dep_terms.extracted_spectra_adu,
                extracted_errors_adu=instrument_dep_terms.extracted_errors_adu,
                extraction_method=instrument_dep_terms.extraction_method,
            )

            ipdb.set_trace()

        # cubes of extracted spectra (wavelength, readout, channel), one file per (QE,
        # DC)
        logger.info("Assembling extracted-spectra cubes ...")
        # ( reads every angle_*_qe_<qe>.hdf5 (one at a time, sorted by angle) and
        #   stacks them; returns the written paths )
        # FYI out: {save_dir}extracted_cube_qe_<qe>_dc_<dc>.fits per DC value:
        #          PRIMARY total (n_bins, n_readouts_total, 4) ADU, one HDU per
        #          component, DARK_CURRENT, ERROR, WAVELENGTH and READOUTS tables;
        #          log "Wrote extracted-spectra cube ..., shape ..." per file;
        #          WARNING "No angle HDF5 files for QE ..." if none found
        cube_file_names = assemble_extracted_spectra_cubes(
            read_dir=str(config["dirs"]["save_s2n_data_unique_dir"]),
            qe=qe,
            save_dir=str(config["dirs"]["save_s2n_data_unique_dir"]),
        )

        ipdb.set_trace()

        save_dir_abs = os.path.abspath(str(config["dirs"]["save_s2n_data_unique_dir"]))
        logger.info(
            "Detector outputs for QE %.2f in %s:\n"
            "  full-frame example readouts: %s\n"
            "  readout cutouts (one per angle): %s\n"
            "  extracted-spectra cubes: %s",
            qe,
            save_dir_abs,
            (
                format_qe_fits_full_example_name(qe)
                if run_limits["write_full_frame_example"]
                else "(not written)"
            ),
            "readouts_angle_*_qe_" + f"{qe:04.2f}.fits",
            ", ".join(os.path.basename(f) for f in cube_file_names) or "none",
        )

    else:
        logger.info("Skipping simulation ...")

    if calculate_s2n:
        # S/N reads the chopped (output 3 - output 4) tables, which are removed for now
        logger.warning(
            "[tasks] calculate_s2n_post_rotation is enabled, but S/N is disabled while "
            "chopping is removed; skipping calculate_s2n_post_rotation"
        )
    else:
        logger.info("Not calculating S/N from HDF5 files.")

    return True


def batch_qe_nint_process(
    base_config_path: str,
    qe_values: List[float],
    sources_to_include: List[str],
    base_filename: str = "s2n",
    overwrite: bool = True,
    plot: bool = False,
    system_params: Optional[dict | pd.Series] = None,
    lum_types: Optional[dict] = None,
    output_root: Optional[str] = None,
    smoke_test: bool = False,
) -> bool:
    """
    Run batch processing with multiple QE and n_int values.

    Args:
        base_config_path: Path to the base configuration file
        n_int_values: List of n_int values to process
        qe_values: List of qe values to process
        sources_to_include: List of sources to include in calculations
        base_filename: Base filename for output files (without extension)
        overwrite: Whether to overwrite existing files
        plot: Whether to generate plots
        system_params: Optional dictionary of the planetary system parameters
        lum_types: Optional dictionary mapping the luminosities to the stellar types
        smoke_test: force smoke-test mode for every run

    Returns:
        True if run_single_calculation succeeded for every QE value
    """
    # Create output directory if it doesn't exist
    # Path(output_dir).mkdir(parents=True, exist_ok=True)

    # initialize for checking if all calculations for the QE, n_int run were successful
    success_all = True

    # make a manual mask over the star?
    override_stellar_mask = bool(True)
    # if override_stellar_mask:
    #     try:
    #         raw_response = input(
    #             "! ------ Inserting a manual stellar mask for ALL outputs. Proceed? (n
    # -> discard manual mask) ------- ! "
    #         )
    #     except EOFError:
    #         raw_response = "y"
    #     response = str(raw_response).strip().lower()
    #     override_stellar_mask = response in {"y", "yes", ""}
    #

    for qe in qe_values:
        # Create output filename
        logging.info("--------------------------------")
        logging.info("--------------------------------")
        logging.info("Processing single calculation:")
        logging.info("Parameter qe = %s", qe)

        # run single calculation, for range of rotation angles
        # this could involve either or both of:
        # 1. generating simulations and writing out the HDF5 files
        # 2. calculating the S/N from the HDF5 files
        time_0 = time.time()
        # ( the whole per-QE pipeline: temp configs -> spectra -> scene -> screens ->
        #   aperture -> detector -> readouts -> extraction -> cubes; returns True )
        # FYI out: everything listed in run_single_calculation (temp .ini files,
        #          readout FITS, angle HDF5, extracted-cube FITS, FYI plots)
        success = run_single_calculation(
            config_path=base_config_path,
            base_filename=base_filename,
            sources_to_include=sources_to_include,
            qe=qe,
            override_stellar_mask=override_stellar_mask,
            overwrite=overwrite,
            plot=plot,
            system_params=system_params,
            lum_types=lum_types,
            output_root=output_root,
            smoke_test=smoke_test,
        )
        time_1 = time.time()
        logging.info(
            "Time taken for run_single_calculation(): %.2f seconds", time_1 - time_0
        )

        if success:
            logging.info("  ✓ Success for the following planetary system parameters:")
            logging.info(system_params)
        else:
            logging.info("  ✗ Failed for qe=%s", qe)
        success_all = success_all and bool(
            success
        )  # was this calculation successful too?

    return success_all


def parameter_sweep(
    config_single_obs_path: str,
    config_sweep_path: str,
    config_planet_population_path: str,
    output_root: Optional[str] = None,
    smoke_test: bool = False,
) -> None:
    """
    Run a QE/n_int parameter sweep for one system or a planet population.

    smoke_test forces smoke-test mode ([smoke_test] in the single-obs config): every
    run is shrunk by its overrides, and the QE values and planets are capped.
    """
    # write a log file even when not called via batch_run.py (no-op if one is set up)
    # ( adds a file handler to the root logger if none writes to a real file )
    # FYI out: sim_pipeline/logs/detectorsim_<timestamp>_<tag>.log (only if new)
    loader.ensure_logging()

    print("\nExample 3: Parameter sweep")
    print("-" * 40)

    # load in all config files once, and don't load them again
    # ( parse each config into a dict and create the directories in its [dirs] )
    # FYI out: log lines "[section]" / "  key = value" for each of the three
    #          configs, "Created directory: ..." per [dirs] entry
    config_single_obs = loader.load_config(config_file=config_single_obs_path)
    config_sweep = loader.load_config(config_file=config_sweep_path)
    config_planet_population = loader.load_config(
        config_file=config_planet_population_path
    )
    lum_types = None

    # ( smoke-test on/off, and its run-size limits; nothing written )
    smoke_test = smoke_test_enabled(config_single_obs, force=smoke_test)
    run_limits = smoke_test_limits(config_single_obs, force=smoke_test)
    if smoke_test:
        logging.warning(
            "*** SMOKE TEST MODE *** (not a science run); limits: %s", run_limits
        )

    # delete preexisting HDF5 output files?
    # ## TODO: make this work; will need to enable deletion of HdF5 files in each output
    # dir determined by the temp config file
    #     generate_sims = loader.config_getboolean(config_single_obs, "tasks",
    # "generate_sims")
    #     if generate_sims:
    #         save_dir = Path(config['dirs']['save_s2n_data_unique_dir'])
    #         existing_hdf5 = sorted(save_dir.glob("angle_*.hdf5"))
    #         if existing_hdf5:
    #             print(f"
    # Found {len(existing_hdf5)} HDF5 file(s) in {save_dir}:")
    #             for p in existing_hdf5[:5]:
    #                 print(f"  - {p.name}")
    #             if len(existing_hdf5) > 5:
    #                 print(f"  ... and {len(existing_hdf5) - 5} more")
    #             try:
    #                 raw = input("Delete existing angle_*.hdf5 files before running?
    # (y/N): ")
    #             except EOFError:
    #                 raw = "n"
    #             if str(raw).strip().lower() in {"y", "yes"}:
    #                 for p in existing_hdf5:
    #                     p.unlink()
    #                 logger.info(f"Deleted {len(existing_hdf5)} HDF5 file(s) from
    # {save_dir}")
    #             else:
    #                 logger.info("Keeping existing HDF5 files")
    #

    if output_root:
        # ( absolute path with trailing separator; for the log line only )
        logging.info(
            "Top-level batch output root override: %s",
            _normalize_output_root(output_root),
        )

    aperture_array_config_path = config_single_obs["telescope"][
        "aperture_array_config_file_name"
    ]
    with open(aperture_array_config_path, "r", encoding="utf-8") as file:
        aperture_array_definition = yaml.safe_load(file)
    n_apertures = len(aperture_array_definition["apertures"])
    logging.info("Number of apertures: %s", n_apertures)

    # calculate the total collecting area of the telescope
    # ( n_apertures x pi (D/2)^2 from the aperture YAML and single_mirror_diameter;
    #   used only for the log line )
    # FYI out: log "Total collecting area of telescope: ..."
    collecting_area = compute_collecting_area_m2(config_single_obs) * u.m**2
    logging.info("Total collecting area of telescope: %s", collecting_area)

    # does the user want to look at a single planet or a planet population?
    systems_2_look_at = config_single_obs["system_options"]["systems_2_look_at"]
    if systems_2_look_at == "planet_population":
        logging.info("Applying parameter sweep to an entire planet population")
    elif systems_2_look_at == "read_in_one_planet_spectrum":
        logging.info("Applying parameter sweep to a single system")
    else:
        logging.error("Invalid system option: %s", systems_2_look_at)
        return

    # if applying a parameter sweep to every planet in a population
    if systems_2_look_at == "planet_population":
        logging.info("Applying parameter sweep to an entire planet population")

        file_name_planet_population = config_planet_population[
            "file_name_planet_population"
        ]["file_name"]
        logging.info("Reading planet population from %s", file_name_planet_population)
        lum_types = config_planet_population[
            "lum_type"
        ]  # map luminosities with stellar types
        # ( one row per planet; nothing written )
        df_planet_population = pd.read_csv(
            file_name_planet_population, skiprows=1, sep=r"\s+"
        )
        # ( left-joins the PSG *.response file names (by planet ID) onto the
        #   population; NaN where no PSG spectrum exists; nothing written )
        df_planet_population = merge_psg_spectra_to_planet_population(
            df_planet_population, config_planet_population
        )
        if run_limits["max_planets"] is not None:
            df_planet_population = df_planet_population.iloc[
                : run_limits["max_planets"]
            ]
            logging.info(
                "Smoke test: limited to the first %d planet(s)",
                len(df_planet_population),
            )

        plot = loader.config_getboolean(
            config_single_obs, "tasks", "print_fyi_plots", default=False
        )
        if plot:
            cols_to_plot = ["Rp", "Porb", "Mp", "z", "Tp", "ap"]
            fyi_plot_path = config_planet_population["file_name_planet_population"][
                "fyi_plot_name"
            ]
            # ( scatter-matrix plot of the population columns above (random sample
            #   of up to 10000 planets) )
            # FYI out: plot at [file_name_planet_population] fyi_plot_name
            plot_planet_population_sample(
                df_planet_population, cols_to_plot, fyi_plot_path
            )
            logging.info("FYI plot of planet population saved to %s", fyi_plot_path)

    elif systems_2_look_at == "read_in_one_planet_spectrum":
        logging.info(
            "Applying parameter sweep to a single planetary system, with spectrum read "
            "in from file"
        )
        df_planet_population = [None]  # wrap in list for length 1

    else:
        logging.error(
            "Invalid system option (must be 'planet_population' or "
            "'read_in_one_planet_spectrum'): %s",
            systems_2_look_at,
        )
        return

    # parameter sweep
    obs = config_sweep["observation"]

    # ( [qe_start, qe_start+qe_step, ..., qe_stop] from the sweep config )
    qe_values = get_sweep_range(obs, "qe")
    if run_limits["max_qe_values"] is not None:
        qe_values = qe_values[: run_limits["max_qe_values"]]
        logging.info("Smoke test: limited to QE value(s) %s", qe_values)

    # get the astrophysical sources to include from the config file
    sources_to_include = [
        source
        for source, include in config_single_obs["astrophysical_sources_to_use"].items()
        if include in [True, "True", "true", 1, "1"]
    ]

    # loop over all the planets (note there can be multiple planets in a system)
    for pl_num in range(len(df_planet_population)):
        if isinstance(df_planet_population, pd.DataFrame):
            system_params = df_planet_population.iloc[pl_num]
            logging.info(
                "Processing system %s with parameters: %s", pl_num, system_params
            )
            base_filename = f"s2n_sweep_planet_index_{pl_num:07d}"
        else:
            system_params = None
            logging.info(
                "No planet population; doing parameter sweep for a single system"
            )
            base_filename = "s2n_sweep"

        # wraps around calculations for a single set of detector parameters
        # ( runs run_single_calculation once per QE value for this planet; returns
        #   True if all succeeded )
        # FYI out: log "Processing single calculation:", "Parameter qe = ...",
        #          "Time taken for run_single_calculation(): ...", success/failure
        #          per QE; plus all outputs of run_single_calculation
        success_all = batch_qe_nint_process(
            base_config_path=config_single_obs_path,
            qe_values=qe_values,
            sources_to_include=sources_to_include,
            base_filename=base_filename,
            overwrite=True,
            plot=True,
            system_params=system_params,
            lum_types=lum_types,
            output_root=output_root,
            smoke_test=smoke_test,
        )
        logging.info(
            "Parameter sweep completed: all calculations successful = %s", success_all
        )
