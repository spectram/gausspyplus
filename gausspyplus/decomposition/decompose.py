import functools
import os
import pickle
from pathlib import Path

from gausspyplus.definitions.config_file import get_values_from_config_file
from gausspyplus.definitions.checks import BaseChecks
from gausspyplus.utils.output import set_up_logger, say, make_pretty_header
from gausspyplus.definitions.definitions import (
    SettingsDefault,
    SettingsDecomposition,
    SettingsImproveFit,
)


class GaussPyDecompose(SettingsDefault, SettingsDecomposition, BaseChecks):
    """Decompose spectra with GaussPy+."""

    def __init__(self, path_to_pickle_file=None, config_file=""):
        self.path_to_pickle_file = path_to_pickle_file
        self.dirpath_gpy = None

        self.single_prepared_spectrum = None

        if config_file:
            get_values_from_config_file(self, config_file, config_key="decomposition")

    @functools.cached_property
    def dirpath(self):
        # TODO: homogenize attributes self.dirpath_gpy (used here) and self.gpy_dirpath (used in training_set)
        return self.dirpath_gpy if self.dirpath_gpy is not None else Path(self.path_to_pickle_file).parents[1]

    @functools.cached_property
    def decomp_dirname(self):
        (decomp_dirname := Path(self.dirpath, "gpy_decomposed")).mkdir(parents=True, exist_ok=True)
        return decomp_dirname

    @functools.cached_property
    def filename_in(self):
        return Path(self.path_to_pickle_file).stem

    @functools.cached_property
    def logger(self):
        if self.single_prepared_spectrum:
            return False
        return (
            False
            if not self.log_output
            else set_up_logger(
                parentDirname=self.dirpath,
                filename=self.filename_in,
                method="g+_decomposition",
            )
        )

    @functools.cached_property
    def pickled_data(self):
        say(
            f"\npickle load '{Path(self.path_to_pickle_file).name}'...",
            logger=self.logger,
        )
        with open(self.path_to_pickle_file, "rb") as pickle_file:
            return pickle.load(pickle_file, encoding="latin1")

    # TODO: Problem with tests: if improve_fitting is changed from False to True cached_property prevents updating the
    #  dictionary
    # @functools.cached_property
    @property
    def fitting(self) -> SettingsImproveFit:
        return SettingsImproveFit(
            improve_fitting=self.improve_fitting,
            min_fwhm=self.min_fwhm,
            max_fwhm=self.max_fwhm,
            snr=self.snr,
            snr_fit=self.snr_fit,
            significance=self.significance,
            snr_negative=self.snr_negative,
            rchi2_limit=self.rchi2_limit,
            max_amp_factor=self.max_amp_factor,
            refit_neg_res_peak=self.refit_neg_res_peak,
            refit_broad=self.refit_broad,
            refit_blended=self.refit_blended,
            fwhm_factor=self.fwhm_factor,
            separation_factor=self.separation_factor,
            exclude_means_outside_channel_range=self.exclude_means_outside_channel_range,
            min_pvalue=self.min_pvalue,
            max_ncomps=self.max_ncomps,
        )

    def decompose(self):
        if self.single_prepared_spectrum:
            self.testing = True
            self.use_ncpus = 1
            say(message=make_pretty_header("GaussPy decomposition"), logger=self.logger)
            return self.start_decomposition()
        else:
            self.raise_exception_if_attribute_is_none("path_to_pickle_file")
            if "testing" in self.pickled_data.keys():
                self.testing = self.pickled_data["testing"]
                if self.testing:
                    self.use_ncpus = 1
            say(message=make_pretty_header("GaussPy decomposition"), logger=self.logger)
            self.start_decomposition()
            if "batchdecomp_temp.pickle" in os.listdir(os.getcwd()):
                os.remove("batchdecomp_temp.pickle")

    def decomposition_settings(self):
        self.set_attribute_if_none("snr_negative", self.snr)
        self.set_attribute_if_none("snr_fit", self.snr / 2)
        self.set_attribute_if_none("snr_thresh", self.snr)
        self.set_attribute_if_none("snr2_thresh", self.snr)

        string_gausspy = str(
            "\ndecomposition settings:"
            "\nGaussPy:"
            f"\nTwo phase decomposition: {self.two_phase_decomposition}"
            f"\nalpha1: {self.alpha1}"
            f"\nalpha2: {self.alpha2}"
            f"\nSNR1: {self.snr_thresh}"
            f"\nSNR2: {self.snr2_thresh}"
        )
        say(string_gausspy, logger=self.logger)

        if self.fitting.improve_fitting:
            string_gausspy_plus = "\n" + "\n".join(
                [
                    f"{attribute}: {getattr(self.fitting, attribute)}"
                    for attribute in dir(self.fitting)
                    if not attribute.startswith("__")
                ]
            )
        else:
            string_gausspy_plus = f"\nimprove_fitting: {self.fitting.improve_fitting}"
        say(string_gausspy_plus, logger=self.logger)

    def start_decomposition(self):
        self.raise_exception_if_attribute_is_none("alpha1")
        if self.two_phase_decomposition:
            self.raise_exception_if_attribute_is_none("alpha2")

        self.decomposition_settings()
        say("\ndecomposing data...", logger=self.logger)

        from gausspyplus.decomposition import gp as gp

        decomposer = gp.GaussianDecomposer()  # Load GaussPy
        decomposer.set("use_ncpus", self.use_ncpus)
        decomposer.set("SNR_thresh", self.snr_thresh)
        decomposer.set("SNR2_thresh", self.snr2_thresh)
        decomposer.set("settings_improve_fit", self.fitting)
        decomposer.set("alpha1", self.alpha1)

        if self.testing:
            decomposer.set("verbose", True)
            decomposer.set("plot", True)

        if self.two_phase_decomposition:
            decomposer.set("phase", "two")
            decomposer.set("alpha2", self.alpha2)
        else:
            decomposer.set("phase", "one")

        if self.single_prepared_spectrum:
            return decomposer.batch_decomposition(dct=self.single_prepared_spectrum)

        self.decomposition = decomposer.batch_decomposition(self.path_to_pickle_file)

        self.save_final_results()

        if self.save_initial_guesses:
            self.save_initial_guesses()

    def save_initial_guesses(self):
        say("\npickle dump GaussPy initial guesses...", logger=self.logger)

        filename = f'{self.filename_in}{"" if self.suffix is None else self.suffix}_fit_ini.pickle'
        pathname = os.path.join(self.decomp_dirname, filename)

        dct_initial_guesses = {
            key: self.decomposition[key]
            for key in [
                "N_components_initial",
                "amplitudes_initial",
                "fwhms_initial",
                "means_initial",
            ]
        }

        pickle.dump(dct_initial_guesses, open(pathname, "wb"), protocol=2)
        say(f"'{filename}' in '{self.decomp_dirname}'", task="save", logger=self.logger)

    def save_final_results(self):
        say("\npickle dump GaussPy final results...", logger=self.logger)

        dct_gausspy_settings = {
            "two_phase": self.two_phase_decomposition,
            "alpha1": self.alpha1,
            "snr1_thresh": self.snr_thresh,
            "snr2_thresh": self.snr2_thresh,
        }

        if self.two_phase_decomposition:
            dct_gausspy_settings["alpha2"] = self.alpha2

        dct_final_guesses = {
            key: self.decomposition[key]
            for key in [
                "index_fit",
                "best_fit_rchi2",
                "best_fit_aicc",
                "pvalue",
                "amplitudes_fit",
                "amplitudes_fit_err",
                "fwhms_fit",
                "fwhms_fit_err",
                "means_fit",
                "means_fit_err",
                "log_gplus",
                "N_neg_res_peak",
                "N_blended",
                "N_components",
                "quality_control",
            ]
        }

        dct_final_guesses["gausspy_settings"] = dct_gausspy_settings

        dct_final_guesses["improve_fit_settings"] = self.fitting

        filename = f'{self.filename_in}{"" if self.suffix is None else self.suffix}_fit_fin.pickle'
        pathname = os.path.join(self.decomp_dirname, filename)
        pickle.dump(dct_final_guesses, open(pathname, "wb"), protocol=2)
        say(f"'{filename}' in '{self.decomp_dirname}'", task="save", logger=self.logger)