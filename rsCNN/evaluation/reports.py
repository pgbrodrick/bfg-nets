import logging
import os
from typing import List

import keras
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsCNN.configuration import configs
from rsCNN.data_management import data_core, sequences
from rsCNN.evaluation import comparisons, histories as reports_histories, inputs, networks, results, samples
from rsCNN.experiments import experiments, histories as experiments_histories


plt.switch_backend('Agg')  # Needed for remote server plotting

_logger = logging.getLogger(__name__)

_FILENAME_MODEL_REPORT = 'model_performance.pdf'


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
        return create_model_report(
            self.experiment.model, self.config, self.data_container.training_sequence,
            self.data_container.validation_sequence, self.experiment.history
        )


def create_model_report(
        model: keras.Model,
        config: configs.Config,
        training_sequence: sequences.BaseSequence,
        validation_sequence: sequences.BaseSequence = None,
        history: dict = None
) -> None:
    is_model_trained = 'lr' in history
    filepath_report = os.path.join(config.model_training.dir_out, _FILENAME_MODEL_REPORT)
    with PdfPages(filepath_report) as pdf:
        _logger.info('Plot Summary')
        _add_figures(networks.print_model_summary(model), pdf)
        _logger.info('Plot Training Sequence Figures')
        sampled = samples.Samples(training_sequence, model, config, data_sequence_label='Training')
        _create_model_report_for_sequence(sampled, config, is_model_trained, pdf)
        _logger.info('Plot Validation Sequence Figures')
        sampled = samples.Samples(validation_sequence, model, config, data_sequence_label='Validation')
        _create_model_report_for_sequence(sampled, config, is_model_trained, pdf)
        _logger.info('Plot Model History')
        if history:
            _add_figures(reports_histories.plot_history(history), pdf)


def _create_model_report_for_sequence(
        sampled: samples.Samples,
        config: configs.Config,
        is_model_trained: bool,
        pdf: PdfPages,
) -> None:
    conf = config.model_reporting
    if is_model_trained and config.architecture.output_activation == 'softmax':
        _add_figures(results.print_classification_report(sampled), pdf)
        _add_figures(results.plot_confusion_matrix(sampled), pdf, tight=False)
    _add_figures(inputs.plot_raw_and_transformed_input_samples(
        sampled, max_pages=conf.max_pages_per_figure, max_samples_per_page=conf.max_samples_per_page,
        max_features_per_page=conf.max_features_per_page, max_responses_per_page=conf.max_responses_per_page), pdf)
    _add_figures(results.single_sequence_prediction_histogram(
        sampled, max_responses_per_page=conf.max_responses_per_page), pdf)
    if is_model_trained:
        _add_figures(results.plot_raw_and_transformed_prediction_samples(
            sampled, max_pages=conf.max_pages_per_figure, max_samples_per_page=conf.max_samples_per_page,
            max_features_per_page=conf.max_features_per_page, max_responses_per_page=conf.max_responses_per_page
        ), pdf)
        if conf.network_progression_show_full:
            _add_figures(networks.plot_network_feature_progression(
                sampled, compact=False, max_pages=conf.network_progression_max_pages,
                max_filters=conf.network_progression_max_filters), pdf)
        if conf.network_progression_show_compact:
            _add_figures(networks.plot_network_feature_progression(
                sampled, compact=True, max_pages=conf.network_progression_max_pages,
                max_filters=conf.network_progression_max_filters), pdf)
        if config.architecture.output_activation == 'softmax':
            plotter = results.plot_spatial_categorical_error
        else:
            plotter = results.plot_spatial_regression_error
        _add_figures(plotter(
            sampled, max_pages=conf.max_pages_per_figure, max_responses_per_page=conf.max_responses_per_page), pdf)


def create_model_comparison_report(
        filepath_out: str,
        dirs_histories: List[str] = None,
        paths_histories: List[str] = None
) -> None:
    assert dirs_histories or paths_histories, \
        'Either provide a directory containing model histories or paths to model histories'
    if not paths_histories:
        paths_histories = list()
    if dirs_histories:
        paths_histories.extend(comparisons.walk_directories_for_model_histories(dirs_histories))
        assert len(paths_histories) > 0, 'No model histories found to compare'
    if not os.path.exists(os.path.dirname(filepath_out)):
        os.makedirs(os.path.dirname(filepath_out))
    model_histories = [experiments_histories.load_history(path_history) for path_history in paths_histories]
    with PdfPages(filepath_out) as pdf:
        _add_figures(comparisons.plot_model_loss_comparison(model_histories), pdf)
        _add_figures(comparisons.plot_model_timing_comparison(model_histories), pdf)


def _add_figures(figures: List[plt.Figure], pdf: PdfPages, tight: bool = True) -> None:
    for fig in figures:
        pdf.savefig(fig, bbox_inches='tight' if tight else None)
