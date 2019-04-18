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
    filepath_report = os.path.join(network_config['model']['dir_out'], 'evaluation_report.pdf')
    sampled_train = samples.Samples(train_sequence, model, network_config)
    with PdfPages(filepath_report) as pdf:
        figures = list()
        # Plot model summary
        figures.append(networks.print_model_summary(model))
        # Plot input data
        figures.extend(inputs.plot_raw_and_transformed_input_samples(sampled_train))
        # Plot results
        figures.extend(results.plot_raw_and_transformed_prediction_samples(sampled_train))
        # Plot compact and expanded network feature progression
        figures.extend(networks.visualize_feature_progression(train_sequence, model, compact=True))
        figures.extend(networks.visualize_feature_progression(train_sequence, model))
        # Plot spatial error
        if network_config['architecture_options']['output_activation'] == 'softmax':
            figures.extend(results.plot_spatial_regression_error(sampled_train))
        else:
            figures.extend(results.plot_spatial_regression_error(sampled_train))
        # Plot training and validation sequence
        figures.extend(results.single_sequence_prediction_histogram(model, train_sequence, 'Training'))
        if (validation_sequence is not None):
            figures.extend(results.single_sequence_prediction_histogram(model, validation_sequence, 'Validation'))
        # Model history
        figures.append(history.plot_history(history))
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')

        # TODO
        # weight_visualization
        # visual_stitching_artifact_check
        # quant_stitching_artificat_check
