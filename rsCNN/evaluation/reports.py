import logging
import os
from typing import List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsCNN.configuration import configs
from rsCNN.data_management import data_core
from rsCNN.evaluation import histories, inputs, networks, results, samples
from rsCNN.experiments import experiments


plt.switch_backend('Agg')  # Needed for remote server plotting

_logger = logging.getLogger(__name__)

_FILENAME_MODEL_REPORT = 'model_performance.pdf'


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
            self._add_figures(networks.print_model_summary(self.experiment.model), pdf)
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
        conf = self.config.model_reporting
        if self.is_model_trained and self.config.architecture.output_activation == 'softmax':
            self._add_figures(results.print_classification_report(sampled), pdf)
            self._add_figures(results.plot_confusion_matrix(sampled), pdf, tight=False)
        self._add_figures(inputs.plot_raw_and_transformed_input_samples(
            sampled, max_pages=conf.max_pages_per_figure, max_samples_per_page=conf.max_samples_per_page,
            max_features_per_page=conf.max_features_per_page, max_responses_per_page=conf.max_responses_per_page), pdf)
        self._add_figures(results.single_sequence_prediction_histogram(
            sampled, max_responses_per_page=conf.max_responses_per_page), pdf)
        if self.is_model_trained:
            self._add_figures(results.plot_raw_and_transformed_prediction_samples(
                sampled, max_pages=conf.max_pages_per_figure, max_samples_per_page=conf.max_samples_per_page,
                max_features_per_page=conf.max_features_per_page, max_responses_per_page=conf.max_responses_per_page
            ), pdf)
            if conf.network_progression_show_full:
                self._add_figures(networks.plot_network_feature_progression(
                    sampled, compact=False, max_pages=conf.network_progression_max_pages,
                    max_filters=conf.network_progression_max_filters), pdf)
            if conf.network_progression_show_compact:
                self._add_figures(networks.plot_network_feature_progression(
                    sampled, compact=True, max_pages=conf.network_progression_max_pages,
                    max_filters=conf.network_progression_max_filters), pdf)
            if self.config.architecture.output_activation == 'softmax':
                plotter = results.plot_spatial_categorical_error
            else:
                plotter = results.plot_spatial_regression_error
            self._add_figures(plotter(
                sampled, max_pages=conf.max_pages_per_figure, max_responses_per_page=conf.max_responses_per_page), pdf)

    def _add_figures(self, figures: List[plt.Figure], pdf: PdfPages, tight: bool = True) -> None:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight' if tight else None)
