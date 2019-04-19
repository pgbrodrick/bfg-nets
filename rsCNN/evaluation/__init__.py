import os

import keras
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.evaluation import histories, inputs, networks, results, samples
from rsCNN.networks import experiments


plt.switch_backend('Agg')  # Needed for remote server plotting


# TODO:  add page with printed warnings and errors from log file, especially following line from NanTermination:
#  Batch 0: Invalid loss, terminating training

# TODO:  Phil:  how can we quantify a "spatial" confusion matrix? e.g., coral classifications are correct near sand
#  and incorrect near deep water. Is this something we can generalize for remote sensing problems?


def create_report_from_experiment(experiment: experiments.Experiment):
    return create_report(
        experiment.model, experiment.train_sequence, experiment.validation_sequence, experiment.test_sequence,
        experiment.network_config, experiment.history
    )


def create_report(
        model: keras.Model,
        network_config: dict,
        train_sequence: BaseSequence,
        validation_sequence: BaseSequence = None,
        test_sequence: BaseSequence = None,
        history: dict = None,
) -> None:
    # TODO:  come up with consistent and general defaults for all visualization parameters (e.g., max_pages) and
    #  update the function definitions to match
    # TODO:  add validation and testing sequence plots where it makes sense, e.g., confusion matrix
    filepath_report = os.path.join(network_config['model']['dir_out'], 'evaluation_report.pdf')
    sampled_train = samples.Samples(train_sequence, model, network_config)
    with PdfPages(filepath_report) as pdf:
        figures = list()
        # Plot model summary
        figures.extend(networks.print_model_summary(model))
        # Plot classification error
        if network_config['architecture_options']['output_activation'] == 'softmax':
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
            figures.extend(results.print_classification_report(sampled_train))
            figures.extend(results.plot_spatial_categorical_error(sampled_train))
        else:
            figures.extend(results.plot_spatial_regression_error(sampled_train))
        # Plot training and validation sequence
        figures.extend(results.single_sequence_prediction_histogram(model, train_sequence, 'Training'))
        if (validation_sequence is not None):
            figures.extend(results.single_sequence_prediction_histogram(model, validation_sequence, 'Validation'))
        # Model history
        figures.extend(history.plot_history(history))
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')

        # TODO
        # weight_visualization
        # visual_stitching_artifact_check
        # quant_stitching_artificat_check
