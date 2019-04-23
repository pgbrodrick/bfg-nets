import os
from typing import List

import keras
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
        experiment.model, experiment.train_sequence, experiment.validation_sequence, experiment.test_sequence,
        experiment.network_config, experiment.history
    )


def create_model_report(
        model: keras.Model,
        network_config: dict,
        train_sequence: BaseSequence,
        validation_sequence: BaseSequence = None,
        test_sequence: BaseSequence = None,
        history: dict = None
) -> None:
    # TODO:  come up with consistent and general defaults for all visualization parameters (e.g., max_pages) and
    #  update the function definitions to match
    # TODO:  add validation and testing sequence plots where it makes sense, e.g., confusion matrix
    filepath_report = os.path.join(network_config['model']['dir_out'], _FILENAME_MODEL_REPORT)
    sampled_train = samples.Samples(train_sequence, model, network_config)
    with PdfPages(filepath_report) as pdf:
        figures = list()
        # Plot model summary
        figures.extend(networks.print_model_summary(model))
        # Plot classification error
        if network_config['architecture_options']['output_activation'] == 'softmax':
            figures.extend(results.print_classification_report(sampled_train))
            figures.extend(results.plot_confusion_matrix(sampled_train))
        # Plot input data
        figures.extend(inputs.plot_raw_and_transformed_input_samples(sampled_train))
        # Plot results
        figures.extend(results.plot_raw_and_transformed_prediction_samples(sampled_train))
        # Plot compact and expanded network feature progression
        figures.extend(networks.plot_network_feature_progression(sampled_train, compact=True))
        figures.extend(networks.plot_network_feature_progression(sampled_train, compact=False))
        # Plot spatial error
        if network_config['architecture_options']['output_activation'] == 'softmax':
            figures.extend(results.plot_spatial_categorical_error(sampled_train))
        else:
            figures.extend(results.plot_spatial_regression_error(sampled_train))
        # Plot training and validation sequence
        figures.extend(results.single_sequence_prediction_histogram(model, train_sequence, 'Training'))
        if validation_sequence is not None:
            figures.extend(results.single_sequence_prediction_histogram(model, validation_sequence, 'Validation'))
        # Model history
        if history:
            figures.extend(plot_history(history))
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')


def create_preliminary_model_report_from_experiment(experiment: experiments.Experiment):
    return create_model_report(
        experiment.model, experiment.train_sequence, experiment.validation_sequence, experiment.test_sequence,
        experiment.network_config
    )


def create_preliminary_model_report(
        model: keras.Model,
        network_config: dict,
        train_sequence: BaseSequence,
        validation_sequence: BaseSequence = None,
        test_sequence: BaseSequence = None,
) -> None:
    filepath_report = os.path.join(network_config['model']['dir_out'], _FILENAME_PRELIMINARY_MODEL_REPORT)
    sampled_train = samples.Samples(train_sequence, model, network_config)
    with PdfPages(filepath_report) as pdf:
        figures = list()
        # Plot model summary
        figures.extend(networks.print_model_summary(model))
        # Plot input data
        figures.extend(inputs.plot_raw_and_transformed_input_samples(sampled_train))
        # Plot training and validation sequence
        figures.extend(results.single_sequence_prediction_histogram(model, train_sequence, 'Training'))
        if validation_sequence is not None:
            figures.extend(results.single_sequence_prediction_histogram(model, validation_sequence, 'Validation'))
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')


def create_model_comparison_report(
        dir_out: str,
        dirs_histories: List[str] = None,
        paths_histories: List[str] = None
) -> None:
    assert dirs_histories or paths_histories, \
        'Either provide a directory containing model histories or paths to model histories'
    if not paths_histories:
        paths_histories = list()
    if dirs_histories:
        paths_histories.extend(comparisons.walk_directories_for_model_histories(dirs_histories))
    model_histories = [histories.load_history(os.path.dirname(p), os.path.basename(p)) for p in paths_histories]
    with PdfPages(os.path.join(dir_out, _FILENAME_MODEL_COMPARISON)) as pdf:
        figures = list()
        figures.extend(comparisons.plot_model_loss_comparison(model_histories))
        figures.extend(comparisons.plot_model_timing_comparison(model_histories))
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')
