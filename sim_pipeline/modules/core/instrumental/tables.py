import logging

import numpy as np
import astropy.units as u
from astropy.table import QTable, Column

from ...pipeline_registry import pipeline_stage
from ...utils.helpers.keys import canonical_dc_rate
from ...viz.tables import plot_debug_final_table

#
class TablesMixin:
    def _build_base_astro_table(self, output_channel) -> QTable:
        # builds a table that consolidates astrophysical signals, before including instrumental noise
        qt = QTable()
        qt['bin'] = Column(
            data=np.arange(len(output_channel.bin_centers)),
            description="Wavelength bin index (0-based)",
        )
        qt['center'] = Column(
            data=output_channel.bin_centers,
            description="Wavelength bin center",
        )
        qt['width'] = Column(
            data=output_channel.bin_widths,
            description="Wavelength bin width",
        )
        qt.meta['wavel_bin_edges'] = output_channel.bin_edges
        qt['npix'] = Column(
            data=np.sum(output_channel.detector.footprint_cube, axis=(1, 2)) * u.pix,
            description="Number of detector pixels in wavelength bin footprint",
        )
        for source_name in self.sources_to_include:
            if source_name not in output_channel.astroph_signal:
                continue
            sig = output_channel.astroph_signal[source_name]
            qt[f'astro_{source_name}_ph_s_um'] = Column(
                data=sig['flux_astro_1d_interpolated_ph_sec_um'],
                description=f"{source_name}: photon rate per micron (all pixels in bin)",
            )
            qt[f'astro_{source_name}_ph_s_bin'] = Column(
                data=sig['flux_astro_1d_interpolated_ph_sec_wavel_bin'],
                description=f"{source_name}: photon rate integrated over wavelength bin (all pixels)",
            )
            qt[f'astro_{source_name}_ph_s_pix'] = Column(
                data=sig['flux_astro_1d_interpolated_ph_sec_pixel'],
                description=f"{source_name}: photon rate per pixel (mean over bin footprint)",
            )
        return qt



    @pipeline_stage(
        depends_on=("pass_through_aperture",),
        cluster="intrinsic instrumental noise",
        cluster_color="#4daf4a",
    )
    def calculate_instrinsic_instrumental_noise(self):
        # RN / DC live on DetectorEffect subclasses; this stage only registers them.
        logging.info("Finding instrumental noise sources...")
        self.readnoise_effect.register(self)
        self.dark_current_effect.register(self) 


    @pipeline_stage(
        depends_on=("apply_detector_effects", "calculate_instrinsic_instrumental_noise"),
    )
    def combine_astro_and_instrum_signals(self, plot: bool = False):
        '''
        Combines astrophysical signals and instrumental noise into one table for permutations of
            1) output
            2) dark current
            3) rotation angle

        INPUTS:

        OUTPUTS:
        - none; updates output_channel.tables_by_dark_current[canonical_dc_rate(dc_rate)]
        '''

        t_frame = float(self.config['observation']['t_int_frame']) * u.second
        read_noise_scalar = self.sources_instrum['read_noise_e_pix-1'] # just one value here
        dc_rates = self.sources_instrum['dark_current_e_pix-1_sec-1']  # (n_dc,)
        gain = float(self.config["detector"]["gain"]) * u.electron / u.adu
        e_per_ph = float(self.config["detector"]["e_per_ph"]) * u.electron / u.ph

        # loop over output channels and make a set of tables (one for each value of dark current)
        for output_name, output_channel in self.output_channels.items():

            base = self._build_base_astro_table(output_channel) # build table that includes astrophysical signals
            n_bins = len(output_channel.bin_centers)
            tables_by_dc = {}

            # loop over dark current values
            for dc_rate_this in dc_rates: 
                
                logging.info(f'Combining astrophysical signals and instrumental noise for output {output_name} at dark current {dc_rate_this}')
                qt = QTable(base.copy())
                # metadata (scalar, for bookkeeping)
                #qt.meta['dark_current_rate_e_pix_sec'] = dc_rate
                # instrumental columns: constant across wavelength bins at this DC
                #qt['instrum_dark_current_e_pix_sec'] = dc_rate * np.ones(n_bins)
                #qt['instrum_dark_current_e_pix'] = np.full(n_bins, dc_rate * t_frame)
                #qt['instrum_read_noise_e_pix'] = read_noise * np.ones(n_bins)
                ## ## TODO: enable multiple reads
                qt['t_int_frame'] = Column(
                    data=t_frame,
                    description="Integration time of one frame",
                )
                qt['qe'] = Column(
                    data=float(self.config['detector']['quantum_efficiency']),
                    description="Detector quantum efficiency",
                )
                tables_by_dc[canonical_dc_rate(dc_rate_this.value)] = qt

            output_channel.tables_by_dark_current_orig = tables_by_dc # _orig meaning that we have not modified the units here

        # loop over each of the tables and make a new table that keeps some of the columns for bookkeeping,
        # and then multiplies others by the appropriate factor to get the total signal in ADU
        for output_name, output_channel in self.output_channels.items():

            for dc_rate, table in output_channel.tables_by_dark_current_orig.items():

                # one table of signals for a permutation of 
                # 1) output
                # 2) dark current
                # 3) rotation angle
                qt = table.copy() 

                # 'dark current' pedestal vs RMS: here we store Poisson RMS so dark-subtraction
                # noise can be propagated as if subtraction were already performed
                qt['instrum_dc_rms_adu'] = Column(
                    data=np.sqrt((dc_rate * u.electron / u.pix) * table['npix'] * t_frame).value * u.electron / gain,
                    description="Dark-current Poisson RMS over wavel bin and detector footprint, for one integration (ADU)",
                )
                qt['instrum_rn_rms_adu'] = Column(
                    data=read_noise_scalar * np.sqrt(table['npix']).value * u.pix / gain,
                    description="Read-noise RMS over wavel bin and detector footprint, for one integration (ADU)",
                )

                # astrophysical sources → ADU per integration (includes × t_frame)
                for source_name in self.sources_to_include:
                    astro_sig = output_channel.astroph_signal[source_name]
                    qt[f'astro_{source_name}_adu'] = Column(
                        data=(
                            astro_sig['flux_astro_1d_interpolated_ph_sec_pixel']
                            * table['npix']
                            * t_frame
                            * e_per_ph
                            / gain
                        ),
                        description=f"{source_name}: signal in wavel bin and detector footprint for one integration (ADU)",
                    )

                qt.meta.update(table.meta)

                # store the final table for this permutation of output, dark current, and rotation angle
                output_channel.tables_by_dark_current[canonical_dc_rate(dc_rate)] = qt

                # plot of final signal in the detector
                if plot:  # pragma: no cover
                    file_name_plot = (
                        str(self.config['dirs']['save_s2n_data_unique_dir'])
                        + f"debug_final_table_{output_name}_dc_{dc_rate:.3f}.png"
                    )
                    plot_debug_final_table(
                        qt,
                        output_channel.bin_edges,
                        output_name,
                        dc_rate,
                        self.sources_to_include,
                        file_name_plot,
                    )
        # for pickling data for Kira
        '''
        out = Path("/Users/eckhartspalding/Downloads/kira/output_channels.pkl")
        payload = {
            name: ch.tables_by_dark_current[0.2]
            for name, ch in self.output_channels.items()
        }
        with out.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        ipdb.set_trace()
        '''
         

    @pipeline_stage(depends_on=("combine_astro_and_instrum_signals",))
    def chop_signal(self, plot: bool = False):
        '''
        Subtracts the dark 3 and 4 outputs

        INPUTS:

        OUTPUTS:
        - None; makes a new chopped table self.post_chop_tables_by_dark_current[dc_rate]
        '''

        self.post_chop_tables_by_dark_current = {}
        for dc_rate, t3 in self.output_channels['output_3_dark'].tables_by_dark_current.items():
            t1 = self.output_channels['output_1_bright'].tables_by_dark_current[dc_rate]
            t2 = self.output_channels['output_2_bright'].tables_by_dark_current[dc_rate]
            t4 = self.output_channels['output_4_dark'].tables_by_dark_current[dc_rate]
            chopped = QTable()

            # copy wavelength metadata once
            for col in ('bin', 'center', 'width', 'npix'):
                chopped[col] = Column(
                    data=t3[col],
                    description=t3[col].info.description,
                )


            # keep outputs, but add the chopped signal
            # consolidate signals from dark outputs, and the chopped signal
            for col in t3.colnames:
                if col.startswith('astro_') and col.endswith('_adu'):
                    chopped[f'output_1_bright_{col}'] = t1[col]
                    chopped[f'output_2_bright_{col}'] = t2[col]
                    chopped[f'output_3_dark_{col}'] = t3[col]
                    chopped[f'output_4_dark_{col}'] = t4[col]
                    chopped[f'chopped_{col}'] = Column(
                        data=t3[col] - t4[col],
                        description=f"Chopped (output_3 − output_4) for {col}",
                    )
                if col in ('instrum_dc_rms_adu', 'instrum_rn_rms_adu'):
                    chopped[f'output_3_dark_{col}'] = t3[col]
                    chopped[f'output_4_dark_{col}'] = t4[col]

                     ## ## TODO: MAKE SURE THIS IS CORRECT
                    chopped[f'chopped_{col}'] = Column(
                        data=np.sqrt(t3[col]**2 + t4[col]**2),
                        description=f"Combined dark-pair RMS for {col}: sqrt(out3^2 + out4^2)",
                    )

            chopped.meta.update(t3.meta)

            self.post_chop_tables_by_dark_current[dc_rate] = chopped
