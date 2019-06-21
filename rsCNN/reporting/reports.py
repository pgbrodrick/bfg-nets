import logging
import os
from typing import List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsCNN.configuration import configs
from rsCNN.data_management import data_core
from rsCNN.reporting import samples
from rsCNN.reporting.visualizations import data_inputs, histories, model_outputs, model_performance, networks
from rsCNN.experiments import experiments


plt.switch_backend('Agg')  # Needed for remote server plotting

_logger = logging.getLogger(__name__)

_FILENAME_MODEL_REPORT = 'model_report.pdf'


class Reporter(object):
    data_container = None
    experiment = None
    config = None
    is_model_trained = None

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
        self.is_model_trained = 'lr' in self.experiment.history

    def create_model_report(self) -> None:
        filepath_report = os.path.join(self.config.model_training.dir_out, _FILENAME_MODEL_REPORT)
        with PdfPages(filepath_report) as pdf:
            _logger.info('Plot Summary')
            self._add_figures(self.plot_model_summary(), pdf)
            _logger.info('Plot Training Sequence Figures')
            sampled = samples.Samples(
                self.data_container.training_sequence, self.experiment.model, self.config,
                data_sequence_label='Training'
            )
            self._create_model_report_for_sequence(sampled, pdf)
            _logger.info('Plot Validation Sequence Figures')
            sampled = samples.Samples(
                self.data_container.validation_sequence, self.experiment.model, self.config,
                data_sequence_label='Validation'
            )
            self._create_model_report_for_sequence(sampled, pdf)
            _logger.info('Plot Model History')
            if self.is_model_trained:
                self._add_figures(histories.plot_history(self.experiment.history), pdf)

    def _create_model_report_for_sequence(self, sampled: samples.Samples, pdf: PdfPages) -> None:
        if self.is_model_trained and self.config.architecture.output_activation == 'softmax':
            self._add_figures(self.plot_classification_report(sampled), pdf)
            self._add_figures(self.plot_confusion_matrix(sampled), pdf, tight=False)
        self._add_figures(self.plot_data_input_samples(sampled), pdf)
        self._add_figures(self.plot_single_sequence_prediction_histogram(sampled), pdf)
        if self.is_model_trained:
            self._add_figures(self.plot_model_output_samples(sampled), pdf)
            if self.config.model_reporting.network_progression_show_full:
                self._add_figures(self.plot_network_feature_progression(sampled, compact=False), pdf)
            if self.config.model_reporting.network_progression_show_compact:
                self._add_figures(self.plot_network_feature_progression(sampled, compact=True), pdf)
            self._add_figures(self.plot_spatial_error(sampled), pdf)

    def _add_figures(self, figures: List[plt.Figure], pdf: PdfPages, tight: bool = True) -> None:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight' if tight else None)

    def plot_confusion_matrix(self, sampled: samples.Samples) -> List[plt.Figure]:
        assert self.is_model_trained, 'Cannot plot confusion matrix because model is not trained.'
        return model_performance.plot_confusion_matrix(sampled)

    def plot_model_summary(self) -> List[plt.Figure]:
        return networks.plot_model_summary(self.experiment.model)

    def plot_model_history(self):
        assert self.is_model_trained, 'Cannot plot model history because model is not trained.'
        return histories.plot_history(self.experiment.history)

    def plot_network_feature_progression(
            self,
            sampled: samples.Samples,
            compact: bool,
            max_pages: int = None,
            max_filters: int = None
    ) -> List[plt.Figure]:
        assert self.is_model_trained, 'Cannot plot network feature progression because model is not trained.'
        return networks.plot_network_feature_progression(
            sampled,
            compact=compact,
            max_pages=max_pages or self.config.model_reporting.network_progression_max_pages,
            max_filters=max_filters or self.config.model_reporting.network_progression_max_filters
        )

    def plot_data_input_samples(
            self,
            sampled: samples.Samples,
            max_pages: int = None,
            max_samples_per_page: int = None,
            max_features_per_page: int = None,
            max_responses_per_page: int = None
    ) -> List[plt.Figure]:
        return data_inputs.plot_data_input_samples(
            sampled,
            max_pages=max_pages or self.config.model_reporting.max_pages_per_figure,
            max_samples_per_page=max_samples_per_page or self.config.model_reporting.max_samples_per_page,
            max_features_per_page=max_features_per_page or self.config.model_reporting.max_features_per_page,
            max_responses_per_page=max_responses_per_page or self.config.model_reporting.max_responses_per_page
        )

    def plot_model_output_samples(
            self,
            sampled: samples.Samples,
            max_pages: int = None,
            max_samples_per_page: int = None,
            max_features_per_page: int = None,
            max_responses_per_page: int = None
    ) -> List[plt.Figure]:
        assert self.is_model_trained, 'Cannot plot raw and transformed prediction samples because model is not trained.'
        return model_outputs.plot_model_output_samples(
            sampled,
            max_pages=max_pages or self.config.model_reporting.max_pages_per_figure,
            max_samples_per_page=max_samples_per_page or self.config.model_reporting.max_samples_per_page,
            max_features_per_page=max_features_per_page or self.config.model_reporting.max_features_per_page,
            max_responses_per_page=max_responses_per_page or self.config.model_reporting.max_responses_per_page
        )

    def plot_single_sequence_prediction_histogram(
            self, sampled: samples.Samples, max_responses_per_page: int = None
    ) -> List[plt.Figure]:
        return model_outputs.plot_single_sequence_prediction_histogram(
            sampled,
            max_responses_per_page=max_responses_per_page or self.config.model_reporting.max_responses_per_page
        )

    def plot_spatial_error(
            self,
            sampled: samples.Samples,
            max_pages: int = None,
            max_responses_per_page: int = None
    ) -> List[plt.Figure]:
        assert self.is_model_trained, 'Cannot plot spatial error because model is not trained.'
        if self.config.architecture.output_activation == 'softmax':
            plotter = model_performance.plot_spatial_categorical_error
        else:
            plotter = model_performance.plot_spatial_regression_error
        return plotter(
            sampled,
            max_pages=max_pages or self.config.model_reporting.max_pages_per_figure,
            max_responses_per_page=max_responses_per_page or self.config.model_reporting.max_responses_per_page
        )

    def plot_classification_report(self, sampled: samples.Samples) -> List[plt.Figure]:
        assert self.is_model_trained, 'Cannot plot classification report because model is not trained.'
        return model_performance.plot_classification_report(sampled)
