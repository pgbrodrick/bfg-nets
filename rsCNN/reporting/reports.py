import logging
import os
from typing import List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsCNN.configuration import configs
from rsCNN.data_management import data_core
from rsCNN.experiments import experiments
from rsCNN.reporting import samples
from rsCNN.reporting.visualizations import histories, logs, model_performance, networks, samples as samples_viz


plt.switch_backend('Agg')  # Needed for remote server plotting

_logger = logging.getLogger(__name__)

_FILENAME_MODEL_REPORT = 'model_report.pdf'

_LABEL_CATEGORICAL = 'CATEGORICAL'
_LABEL_CONTINUOUS = 'CONTINUOUS'


class Reporter(object):
    data_container = None
    experiment = None
    config = None

    def __init__(
            self,
            data_container: data_core.DataContainer,
            experiment: experiments.Experiment,
            config: configs.Config
    ) -> None:
        errors = config.get_human_readable_config_errors(include_sections=['model_reporting'])
        assert not errors, errors
        self.data_container = data_container
        self.experiment = experiment
        self.config = config

    def create_model_report(self) -> None:
        filepath_report = os.path.join(self.config.model_training.dir_out, _FILENAME_MODEL_REPORT)
        with PdfPages(filepath_report) as pdf:
            _logger.info('Plot Summary')
            self._add_figures(self.plot_model_summary(), pdf)
            _logger.info('Plot Training Sequence Figures')
            sampled = samples.Samples(
                self.data_container.training_sequence, self.experiment.model, self.config,
                self.experiment.is_model_trained, data_sequence_label='Training'
            )
            self._create_model_report_for_sequence(sampled, pdf)
            _logger.info('Plot Validation Sequence Figures')
            sampled = samples.Samples(
                self.data_container.validation_sequence, self.experiment.model, self.config,
                self.experiment.is_model_trained, data_sequence_label='Validation'
            )
            self._create_model_report_for_sequence(sampled, pdf)
            _logger.info('Plot Model History')
            if self.experiment.is_model_trained:
                self._add_figures(histories.plot_history(self.experiment.history), pdf)
            if self.experiment.is_model_trained:
                self._add_figures(logs.plot_log_warnings_and_errors(
                    experiments.get_log_filepath(self.config.model_training.dir_out)), pdf)

    def _create_model_report_for_sequence(self, sampled: samples.Samples, pdf: PdfPages) -> None:
        if self.experiment.is_model_trained and self._get_response_data_types() is _LABEL_CATEGORICAL:
            self._add_figures(self.plot_classification_report(sampled), pdf)
            self._add_figures(self.plot_confusion_matrix(sampled), pdf, tight=False)
        self._add_figures(self.plot_single_sequence_prediction_histogram(sampled), pdf)
        if self.experiment.is_model_trained:
            self._add_figures(self.plot_samples(sampled), pdf)
            if self.config.model_reporting.network_progression_show_full:
                self._add_figures(self.plot_network_feature_progression(sampled, compact=False), pdf)
            if self.config.model_reporting.network_progression_show_compact:
                self._add_figures(self.plot_network_feature_progression(sampled, compact=True), pdf)
            self._add_figures(self.plot_spatial_error(sampled), pdf)

    def _add_figures(self, figures: List[plt.Figure], pdf: PdfPages, tight: bool = True) -> None:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight' if tight else None)

    def plot_confusion_matrix(self, sampled: samples.Samples) -> List[plt.Figure]:
        assert self.experiment.is_model_trained, 'Cannot plot confusion matrix because model is not trained.'
        return model_performance.plot_confusion_matrix(sampled)

    def plot_model_summary(self) -> List[plt.Figure]:
        return networks.plot_model_summary(self.experiment.model)

    def plot_model_history(self):
        assert self.experiment.is_model_trained, 'Cannot plot model history because model is not trained.'
        return histories.plot_history(self.experiment.history)

    def plot_network_feature_progression(
            self,
            sampled: samples.Samples,
            compact: bool,
            max_pages: int = None,
            max_filters: int = None
    ) -> List[plt.Figure]:
        assert self.experiment.is_model_trained, 'Cannot plot network feature progression because model is not trained.'
        return networks.plot_network_feature_progression(
            sampled,
            compact=compact,
            max_pages=max_pages or self.config.model_reporting.network_progression_max_pages,
            max_filters=max_filters or self.config.model_reporting.network_progression_max_filters
        )

    def plot_samples(
            self,
            sampled: samples.Samples,
            max_pages: int = None,
            max_samples_per_page: int = None,
            max_features_per_page: int = None,
            max_responses_per_page: int = None
    ) -> List[plt.Figure]:
        assert self.experiment.is_model_trained, \
            'Cannot plot raw and transformed prediction samples because model is not trained.'
        if self._get_response_data_types() is _LABEL_CATEGORICAL:
            plotter = samples_viz.plot_classification_samples
        elif self._get_response_data_types() is _LABEL_CONTINUOUS:
            plotter = samples_viz.plot_regression_samples
        return plotter(
            sampled,
            max_pages=max_pages or self.config.model_reporting.max_pages_per_figure,
            max_samples_per_page=max_samples_per_page or self.config.model_reporting.max_samples_per_page,
            max_features_per_page=max_features_per_page or self.config.model_reporting.max_features_per_page,
            max_responses_per_page=max_responses_per_page or self.config.model_reporting.max_responses_per_page
        )

    def plot_single_sequence_prediction_histogram(
            self, sampled: samples.Samples, max_responses_per_page: int = None
    ) -> List[plt.Figure]:
        return samples_viz.plot_single_sequence_prediction_histogram(
            sampled,
            max_responses_per_page=max_responses_per_page or self.config.model_reporting.max_responses_per_page
        )

    def plot_spatial_error(
            self,
            sampled: samples.Samples,
            max_pages: int = None,
            max_responses_per_page: int = None
    ) -> List[plt.Figure]:
        assert self.experiment.is_model_trained, 'Cannot plot spatial error because model is not trained.'
        if self._get_response_data_types() is _LABEL_CATEGORICAL:
            plotter = model_performance.plot_spatial_classification_error
        elif self._get_response_data_types() is _LABEL_CONTINUOUS:
            plotter = model_performance.plot_spatial_regression_error
        return plotter(
            sampled,
            max_pages=max_pages or self.config.model_reporting.max_pages_per_figure,
            max_responses_per_page=max_responses_per_page or self.config.model_reporting.max_responses_per_page
        )

    def plot_classification_report(self, sampled: samples.Samples) -> List[plt.Figure]:
        assert self.experiment.is_model_trained, 'Cannot plot classification report because model is not trained.'
        return model_performance.plot_classification_report(sampled)

    def _get_response_data_types(self) -> str:
        data_types = set([data_type for file_dt in self.config.raw_files.response_data_type for data_type in file_dt])
        if data_types == {'C'}:
            return _LABEL_CATEGORICAL
        elif data_types == {'R'}:
            return _LABEL_CONTINUOUS
        elif data_types == {'C', 'R'}:
            raise AssertionError('Reporter does not currently support mixed response data types.')
        else:
            raise AssertionError('Unexpected data types found: {}.'.format(data_types))
