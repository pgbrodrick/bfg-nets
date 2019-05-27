import os
from typing import List

import keras
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsCNN.configuration import configs
from rsCNN.data_management.sequences import BaseSequence
from rsCNN.evaluation import comparisons, inputs, networks, results, samples
from rsCNN.evaluation.histories import plot_history
from rsCNN.networks import experiments, histories


plt.switch_backend('Agg')  # Needed for remote server plotting


_FILENAME_MODEL_COMPARISON = 'model_comparison.pdf'
_FILENAME_MODEL_REPORT = 'model_performance.pdf'
_FILENAME_PRELIMINARY_MODEL_REPORT = 'model_overview.pdf'


# TODO:  add page with printed warnings and errors from log file, especially following line from NanTermination:
#  Batch 0: Invalid loss, terminating training

# TODO:  Phil:  how can we quantify a "spatial" confusion matrix? e.g., coral classifications are correct near sand
#  and incorrect near deep water. Is this something we can generalize for remote sensing problems?


def create_model_report_from_experiment(experiment: experiments.Experiment):
    return create_model_report(
        experiment.model, experiment.config, experiment.train_sequence, experiment.validation_sequence,
        experiment.history
    )


def create_model_report(
        model: keras.Model,
        config: configs.Config,
        train_sequence: BaseSequence,
        validation_sequence: BaseSequence = None,
        history: dict = None
) -> None:
    # TODO:  come up with consistent and general defaults for all visualization parameters (e.g., max_pages) and
    #  update the function definitions to match
    filepath_report = os.path.join(config.model_training.dir_out, _FILENAME_MODEL_REPORT)
    with PdfPages(filepath_report) as pdf:
        # Plot model summary
        _add_figures(networks.print_model_summary(model), pdf)
        # Plot training sequence figures
        sampled = samples.Samples(train_sequence, model, config, data_sequence_label='Training')
        if config.architecture.output_activation == 'softmax':
            _add_figures(results.print_classification_report(sampled), pdf)
            _add_figures(results.plot_confusion_matrix(sampled), pdf)
        _add_figures(inputs.plot_raw_and_transformed_input_samples(sampled), pdf)
        _add_figures(results.single_sequence_prediction_histogram(sampled), pdf)
        _add_figures(results.plot_raw_and_transformed_prediction_samples(sampled), pdf)
        _add_figures(networks.plot_network_feature_progression(sampled, compact=True), pdf)
        _add_figures(networks.plot_network_feature_progression(sampled, compact=False), pdf)
        if config.architecture.output_activation == 'softmax':
            _add_figures(results.plot_spatial_categorical_error(sampled), pdf)
        else:
            _add_figures(results.plot_spatial_regression_error(sampled), pdf)
        # Plot validation sequence figures
        if validation_sequence is not None:
            sampled = samples.Samples(validation_sequence, model, config, data_sequence_label='Validation')
            if config.architecture.output_activation == 'softmax':
                _add_figures(results.print_classification_report(sampled), pdf)
                _add_figures(results.plot_confusion_matrix(sampled), pdf)
            _add_figures(inputs.plot_raw_and_transformed_input_samples(sampled), pdf)
            _add_figures(results.single_sequence_prediction_histogram(sampled), pdf)
            _add_figures(results.plot_raw_and_transformed_prediction_samples(sampled), pdf)
            _add_figures(networks.plot_network_feature_progression(sampled, compact=True), pdf)
            _add_figures(networks.plot_network_feature_progression(sampled, compact=False), pdf)
            if config.architecture.output_activation == 'softmax':
                _add_figures(results.plot_spatial_categorical_error(sampled), pdf)
            else:
                _add_figures(results.plot_spatial_regression_error(sampled), pdf)
        # Model history
        if history:
            _add_figures(plot_history(history), pdf)


def create_preliminary_model_report_from_experiment(experiment: experiments.Experiment):
    return create_preliminary_model_report(
        experiment.model, experiment.config, experiment.train_sequence, experiment.validation_sequence
    )


def create_preliminary_model_report(
        model: keras.Model,
        config: configs.Config,
        train_sequence: BaseSequence,
        validation_sequence: BaseSequence = None,
) -> None:
    # TODO:  combine with other model report function, just have if statements to avoid plots that can't be created
    filepath_report = os.path.join(config.model_training.dir_out, _FILENAME_MODEL_REPORT)
    with PdfPages(filepath_report) as pdf:
        # Plot model summary
        _add_figures(networks.print_model_summary(model), pdf)
        # Plot training sequence figures
        sampled = samples.Samples(train_sequence, model, config, data_sequence_label='Training')
        _add_figures(inputs.plot_raw_and_transformed_input_samples(sampled), pdf)
        _add_figures(results.single_sequence_prediction_histogram(sampled), pdf)
        del sampled
        # Plot validation sequence figures
        if validation_sequence is not None:
            sampled = samples.Samples(train_sequence, model, config, data_sequence_label='Validation')
            _add_figures(inputs.plot_raw_and_transformed_input_samples(sampled), pdf)
            _add_figures(results.single_sequence_prediction_histogram(sampled), pdf)
            del sampled


def create_model_comparison_report(
        dir_out: str,
        filename: str = None,
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
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    model_histories = [histories.load_history(os.path.dirname(p), os.path.basename(p)) for p in paths_histories]
    with PdfPages(os.path.join(dir_out, filename or _FILENAME_MODEL_COMPARISON)) as pdf:
        _add_figures(comparisons.plot_model_loss_comparison(model_histories), pdf)
        _add_figures(comparisons.plot_model_timing_comparison(model_histories), pdf)


def _add_figures(figures: List[plt.Figure], pdf: PdfPages) -> None:
    for fig in figures:
        pdf.savefig(fig, bbox_inches='tight')
